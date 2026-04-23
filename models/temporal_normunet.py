from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from cardiac_ensure.ops import dynamic_a_normal


EPS = 1e-8


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        drop_prob: float,
        activate: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
        ]
        if activate:
            layers.extend(
                [
                    nn.InstanceNorm2d(out_chans),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                ]
            )
            if drop_prob > 0:
                layers.append(nn.Dropout2d(drop_prob))
        self.layers = nn.Sequential(*layers)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.layers(image)


class PaperCnn(nn.Module):
    """
    Paper-style CNN denoiser adapted to the time-folded setting.

    The original paper uses real/imaginary channels (2 channels total). Here we
    keep the existing temporal folding, so the effective input/output channel
    count becomes 2 * num_frames while the hidden width follows the requested
    64-channel shape.
    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        chans: int = 64,
        num_layers: int = 4,
        drop_prob: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")

        layers: list[nn.Module] = []
        current_in = in_chans
        for _ in range(num_layers):
            layers.append(ConvLayer(current_in, chans, drop_prob=drop_prob, activate=True))
            current_in = chans
        layers.append(ConvLayer(current_in, out_chans, drop_prob=drop_prob, activate=False))
        self.layers = nn.Sequential(*layers)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.layers(image)


class NormUnet(nn.Module):
    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int,
        out_chans: int,
        drop_prob: float = 0.0,
    ) -> None:
        super().__init__()
        self.cnn = PaperCnn(
            in_chans=in_chans,
            out_chans=out_chans,
            chans=chans,
            num_layers=num_pools,
            drop_prob=drop_prob,
        )

    def complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width, two = x.shape
        if two != 2:
            raise ValueError(f"Expected last dim 2, got {tuple(x.shape)}")
        return x.permute(0, 4, 1, 2, 3).reshape(batch_size, 2 * channels, height, width)

    def chan_complex_to_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels_twice, height, width = x.shape
        if channels_twice % 2 != 0:
            raise ValueError(f"Expected an even number of channels, got {channels_twice}")
        channels = channels_twice // 2
        return x.view(batch_size, 2, channels, height, width).permute(0, 2, 3, 4, 1).contiguous()

    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, channels, height, width = x.shape
        x_grouped = x.reshape(batch_size, 2, (channels // 2) * height * width)
        mean = x_grouped.mean(dim=2).view(batch_size, 2, 1, 1, 1)
        std = x_grouped.std(dim=2).clamp_min(EPS).view(batch_size, 2, 1, 1, 1)
        x_grouped = x.view(batch_size, 2, channels // 2, height, width)
        x_norm = ((x_grouped - mean) / std).view(batch_size, channels, height, width)
        mean = mean.expand(-1, -1, channels // 2, -1, -1).reshape(batch_size, channels, 1, 1)
        std = std.expand(-1, -1, channels // 2, -1, -1).reshape(batch_size, channels, 1, 1)
        return x_norm, mean, std

    def unnorm(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return x * std + mean

    def pad(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        _, _, height, width = x.shape
        width_mult = ((width - 1) | 15) + 1
        height_mult = ((height - 1) | 15) + 1
        width_pad = [math.floor((width_mult - width) / 2), math.ceil((width_mult - width) / 2)]
        height_pad = [math.floor((height_mult - height) / 2), math.ceil((height_mult - height) / 2)]
        x = F.pad(x, width_pad + height_pad)
        return x, (height_pad, width_pad, height_mult, width_mult)

    def unpad(
        self,
        x: torch.Tensor,
        height_pad: List[int],
        width_pad: List[int],
        height_mult: int,
        width_mult: int,
    ) -> torch.Tensor:
        return x[..., height_pad[0] : height_mult - height_pad[1], width_pad[0] : width_mult - width_pad[1]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or x.shape[-1] != 2:
            raise ValueError(f"NormUnet expects [B, C, H, W, 2], got {tuple(x.shape)}")

        x = self.complex_to_chan_dim(x)
        x, mean, std = self.norm(x)
        x, pad_sizes = self.pad(x)
        x = self.cnn(x)
        x = self.unpad(x, *pad_sizes)
        x = self.unnorm(x, mean, std)
        return self.chan_complex_to_last_dim(x)


class SoftDataConsistency(nn.Module):
    """Basic soft DC step: x <- x - lambda * (A^H A x - x0)."""

    def __init__(self, init_weight: float = 1.0) -> None:
        super().__init__()
        init = min(max(float(init_weight), 1e-4), 1.0 - 1e-4)
        raw_init = math.log(init / (1.0 - init))
        self.dc_weight = nn.Parameter(torch.tensor(raw_init, dtype=torch.float32))

    def dc_weight_value(self) -> torch.Tensor:
        return torch.sigmoid(self.dc_weight)

    def forward(
        self,
        image: torch.Tensor,
        reference_image: torch.Tensor,
        maps: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if maps is None or mask is None:
            return image

        normal = dynamic_a_normal(image, maps, mask)
        dc_weight = self.dc_weight_value().to(device=image.device, dtype=image.real.dtype)
        return image - dc_weight * (normal - reference_image)


def extract_center_frame(x: torch.Tensor, temporal_dim: int = 1) -> torch.Tensor:
    center = x.shape[temporal_dim] // 2
    index = [slice(None)] * x.ndim
    index[temporal_dim] = slice(center, center + 1)
    return x[tuple(index)]


class TemporalNormUnet(nn.Module):
    """Time-folded CNN + soft-DC cascades for dynamic cardiac MRI windows."""

    def __init__(
        self,
        num_frames: int,
        chans: int = 64,
        num_pools: int = 4,
        drop_prob: float = 0.0,
        residual: bool = True,
        output_mode: str = "all_frames",
        num_unrolls: int = 3,
        use_data_consistency: bool = True,
    ) -> None:
        super().__init__()
        if num_frames <= 0:
            raise ValueError(f"num_frames must be positive, got {num_frames}")
        if num_pools <= 0:
            raise ValueError(f"num_pools must be positive, got {num_pools}")
        if num_unrolls <= 0:
            raise ValueError(f"num_unrolls must be positive, got {num_unrolls}")
        if output_mode not in {"all_frames", "center_frame"}:
            raise ValueError(f"Unsupported output_mode={output_mode}")

        self.num_frames = int(num_frames)
        self.center_frame = self.num_frames // 2
        self.residual = bool(residual)
        self.output_mode = output_mode
        self.num_unrolls = int(num_unrolls)
        self.use_data_consistency = bool(use_data_consistency)

        self.norm_unet = NormUnet(
            chans=chans,
            num_pools=num_pools,
            in_chans=2 * self.num_frames,
            out_chans=2 * self.num_frames,
            drop_prob=drop_prob,
        )
        self.dc_blocks = nn.ModuleList(
            [SoftDataConsistency(init_weight=0.99) for _ in range(self.num_unrolls)]
        )

    def _canonicalize_input(self, x: torch.Tensor) -> Tuple[torch.Tensor, bool, bool]:
        input_was_real = False
        input_was_batched = True

        if torch.is_complex(x):
            if x.ndim == 5 and x.shape[2] == 1:
                return x, input_was_real, input_was_batched
            if x.ndim == 4:
                if x.shape[1] == 1:
                    input_was_batched = False
                    return x.unsqueeze(0), input_was_real, input_was_batched
                return x[:, :, None, ...], input_was_real, input_was_batched
            if x.ndim == 3:
                input_was_batched = False
                return x.unsqueeze(0)[:, :, None, ...], input_was_real, input_was_batched
            raise ValueError(
                "Complex input must have shape [B, T, 1, H, W], [T, 1, H, W], [B, T, H, W], or [T, H, W], "
                f"got {tuple(x.shape)}"
            )

        if x.shape[-1] != 2:
            raise ValueError(f"Real input must have trailing complex dim 2, got {tuple(x.shape)}")

        input_was_real = True
        if x.ndim == 4:
            x = x.unsqueeze(0)
            input_was_batched = False
        if x.ndim != 5:
            raise ValueError(f"Real input must have shape [B, T, H, W, 2] or [T, H, W, 2], got {tuple(x.shape)}")
        return torch.view_as_complex(x.contiguous())[:, :, None, ...], input_was_real, input_was_batched

    def _restore_output(self, x: torch.Tensor, input_was_real: bool, input_was_batched: bool) -> torch.Tensor:
        if not input_was_batched:
            x = x[0]

        if input_was_real:
            if input_was_batched:
                x = x.squeeze(2) if x.ndim == 5 and x.shape[2] == 1 else x
            else:
                x = x.squeeze(1) if x.ndim == 4 and x.shape[1] == 1 else x
            return torch.view_as_real(x.contiguous())

        return x

    def _run_denoiser(self, x_complex: torch.Tensor) -> torch.Tensor:
        x_real = torch.view_as_real(x_complex.squeeze(2).contiguous())
        y_real = self.norm_unet(x_real)
        y_complex = torch.view_as_complex(y_real.contiguous())[:, :, None, ...]
        if self.residual:
            y_complex = y_complex + x_complex
        return y_complex

    def forward(
        self,
        x: torch.Tensor,
        maps: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_complex, input_was_real, input_was_batched = self._canonicalize_input(x)
        batch_size, num_frames, channels, _, _ = x_complex.shape
        if channels != 1:
            raise ValueError(f"TemporalNormUnet expects a single complex channel, got {channels}")
        if num_frames != self.num_frames:
            raise ValueError(f"Expected num_frames={self.num_frames}, got {num_frames}")

        current = x_complex
        reference = x_complex
        for cascade_idx in range(self.num_unrolls):
            current = self._run_denoiser(current)
            if self.use_data_consistency:
                current = self.dc_blocks[cascade_idx](current, reference_image=reference, maps=maps, mask=mask)

        if self.output_mode == "center_frame":
            current = extract_center_frame(current, temporal_dim=1)

        return self._restore_output(current, input_was_real, input_was_batched)
