from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.fft as torch_fft


EPS = 1e-8


@dataclass(frozen=True)
class DynamicLayout:
    batch_size: int
    num_frames: int
    num_coils: int
    height: int
    width: int
    squeezed_batch: bool


def fft2c(x: torch.Tensor) -> torch.Tensor:
    x = torch_fft.fftshift(x, dim=(-2, -1))
    x = torch_fft.fft2(x, dim=(-2, -1), norm="ortho")
    x = torch_fft.ifftshift(x, dim=(-2, -1))
    return x


def ifft2c(x: torch.Tensor) -> torch.Tensor:
    x = torch_fft.ifftshift(x, dim=(-2, -1))
    x = torch_fft.ifft2(x, dim=(-2, -1), norm="ortho")
    x = torch_fft.fftshift(x, dim=(-2, -1))
    return x


def _check_complex(name: str, tensor: torch.Tensor) -> None:
    if not torch.is_complex(tensor):
        raise TypeError(f"{name} must be a complex tensor, got {tensor.dtype}")


def _ensure_frame_image(image: torch.Tensor) -> torch.Tensor:
    _check_complex("image", image)
    if image.ndim == 3:
        image = image[:, None, ...]
    if image.ndim != 4 or image.shape[1] != 1:
        raise ValueError(
            "Single-frame image must have shape [N, 1, H, W] or [N, H, W], "
            f"got {tuple(image.shape)}"
        )
    return image


def _ensure_frame_kspace(kspace: torch.Tensor) -> torch.Tensor:
    _check_complex("kspace", kspace)
    if kspace.ndim != 4:
        raise ValueError(f"Single-frame kspace must have shape [N, C, H, W], got {tuple(kspace.shape)}")
    return kspace


def _ensure_frame_maps(maps: torch.Tensor, batch_size: int) -> torch.Tensor:
    _check_complex("maps", maps)
    if maps.ndim == 3:
        maps = maps.unsqueeze(0)
    if maps.ndim != 4:
        raise ValueError(f"maps must have shape [C, H, W] or [N, C, H, W], got {tuple(maps.shape)}")
    if maps.shape[0] == 1 and batch_size > 1:
        maps = maps.expand(batch_size, -1, -1, -1)
    if maps.shape[0] != batch_size:
        raise ValueError(f"maps batch={maps.shape[0]} does not match image batch={batch_size}")
    return maps


def _ensure_frame_mask(
    mask: torch.Tensor,
    batch_size: int,
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    if mask.ndim == 2:
        mask = mask[None, None, ...]
    elif mask.ndim == 3:
        if mask.shape[0] in {1, batch_size}:
            mask = mask[:, None, ...]
        else:
            raise ValueError(f"Ambiguous mask shape {tuple(mask.shape)} for batch={batch_size}")
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError(
            "Mask must have shape [N, 1, H, W], [N, H, W], [1, H, W], or [H, W], "
            f"got {tuple(mask.shape)}"
        )
    if mask.shape[0] == 1 and batch_size > 1:
        mask = mask.expand(batch_size, -1, -1, -1)
    if mask.shape[0] != batch_size or mask.shape[-2:] != (height, width):
        raise ValueError(
            f"Mask shape {tuple(mask.shape)} is incompatible with batch={batch_size}, spatial={(height, width)}"
        )
    return mask.to(device=device, dtype=torch.float32)


def apply_density_weight(kspace: torch.Tensor, density_weight: Optional[torch.Tensor]) -> torch.Tensor:
    if density_weight is None:
        return kspace
    weight = _expand_density_weight(density_weight, kspace)
    return kspace * weight


def _expand_density_weight(density_weight: torch.Tensor, kspace: torch.Tensor) -> torch.Tensor:
    if density_weight is None:
        return torch.ones_like(kspace.real)
    weight = torch.as_tensor(density_weight, device=kspace.device, dtype=kspace.real.dtype)
    batch_size, _, height, width = kspace.shape
    if weight.ndim == 1:
        if weight.shape[0] != width:
            raise ValueError(f"Width-only density weight must have size {width}, got {tuple(weight.shape)}")
        weight = weight.view(1, 1, 1, width)
    elif weight.ndim == 2:
        if tuple(weight.shape) == (height, width):
            weight = weight.view(1, 1, height, width)
        elif tuple(weight.shape) == (batch_size, width):
            weight = weight.view(batch_size, 1, 1, width)
        else:
            raise ValueError(f"Unsupported density-weight shape {tuple(weight.shape)}")
    elif weight.ndim == 3:
        if tuple(weight.shape) == (batch_size, height, width):
            weight = weight.view(batch_size, 1, height, width)
        else:
            raise ValueError(f"Unsupported density-weight shape {tuple(weight.shape)}")
    elif weight.ndim == 4:
        if weight.shape[0] not in {1, batch_size} or weight.shape[1] not in {1, kspace.shape[1]}:
            raise ValueError(f"Unsupported density-weight shape {tuple(weight.shape)}")
    else:
        raise ValueError(f"Unsupported density-weight shape {tuple(weight.shape)}")
    return torch.broadcast_to(weight, kspace.real.shape)


def sense_forward(image: torch.Tensor, maps: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    image = _ensure_frame_image(image)
    batch_size, _, height, width = image.shape
    maps = _ensure_frame_maps(maps, batch_size)
    mask = _ensure_frame_mask(mask, batch_size, height, width, image.device)

    coil_imgs = maps * image
    coil_kspace = fft2c(coil_imgs)
    return coil_kspace * mask


def sense_adjoint(kspace: torch.Tensor, maps: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    kspace = _ensure_frame_kspace(kspace)
    batch_size, _, height, width = kspace.shape
    maps = _ensure_frame_maps(maps, batch_size)
    mask = _ensure_frame_mask(mask, batch_size, height, width, kspace.device)

    sampled_kspace = kspace * mask
    coil_imgs = ifft2c(sampled_kspace)
    img_out = torch.sum(torch.conj(maps) * coil_imgs, dim=1, keepdim=True)
    return img_out


def sense_normal(image: torch.Tensor, maps: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return sense_adjoint(sense_forward(image, maps, mask), maps, mask)


def sense_weighted_forward(
    image: torch.Tensor,
    maps: torch.Tensor,
    mask: torch.Tensor,
    density_weight: Optional[torch.Tensor],
) -> torch.Tensor:
    return apply_density_weight(sense_forward(image, maps, mask), density_weight)


def sense_weighted_adjoint(
    kspace: torch.Tensor,
    maps: torch.Tensor,
    mask: torch.Tensor,
    density_weight: Optional[torch.Tensor],
) -> torch.Tensor:
    return sense_adjoint(apply_density_weight(kspace, density_weight), maps, mask)


def sense_weighted_normal(
    image: torch.Tensor,
    maps: torch.Tensor,
    mask: torch.Tensor,
    density_weight: Optional[torch.Tensor],
) -> torch.Tensor:
    return sense_weighted_adjoint(
        sense_weighted_forward(image, maps, mask, density_weight),
        maps,
        mask,
        density_weight,
    )


def _ensure_dynamic_image(image: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    _check_complex("image", image)
    if image.ndim == 3:
        image = image[:, None, ...]
    if image.ndim == 4 and image.shape[1] == 1:
        return image.unsqueeze(0), True
    if image.ndim == 5 and image.shape[2] == 1:
        return image, False
    raise ValueError(
        "Dynamic image must have shape [T, 1, H, W], [T, H, W], or [B, T, 1, H, W], "
        f"got {tuple(image.shape)}"
    )


def _ensure_dynamic_kspace(kspace: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    _check_complex("kspace", kspace)
    if kspace.ndim == 4:
        return kspace.unsqueeze(0), True
    if kspace.ndim == 5:
        return kspace, False
    raise ValueError(f"Dynamic kspace must have shape [T, C, H, W] or [B, T, C, H, W], got {tuple(kspace.shape)}")


def _ensure_dynamic_maps(
    maps: torch.Tensor,
    batch_size: int,
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    maps = _ensure_frame_maps(maps.to(device=device), batch_size)
    if maps.shape[-2:] != (height, width):
        raise ValueError(f"Map spatial shape {tuple(maps.shape[-2:])} does not match {(height, width)}")
    return maps


def _ensure_dynamic_mask(
    mask: torch.Tensor,
    batch_size: int,
    num_frames: int,
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    if mask.ndim == 2:
        mask = mask[None, None, None, ...]
    elif mask.ndim == 3:
        mask = mask[:, None, ...]
    if mask.ndim == 4 and mask.shape[1] == 1:
        mask = mask.unsqueeze(0)
    elif mask.ndim == 4 and mask.shape[0] in {1, batch_size} and mask.shape[-2:] == (height, width):
        mask = mask[:, :, None, ...]
    if mask.ndim != 5 or mask.shape[2] != 1:
        raise ValueError(
            "Dynamic mask must have shape [T, 1, H, W], [B, T, 1, H, W], or broadcastable variants, "
            f"got {tuple(mask.shape)}"
        )
    if mask.shape[0] == 1 and batch_size > 1:
        mask = mask.expand(batch_size, -1, -1, -1, -1)
    if mask.shape[1] == 1 and num_frames > 1:
        mask = mask.expand(-1, num_frames, -1, -1, -1)
    if mask.shape[:2] != (batch_size, num_frames) or mask.shape[-2:] != (height, width):
        raise ValueError(
            f"Mask shape {tuple(mask.shape)} is incompatible with batch={batch_size}, frames={num_frames}, "
            f"spatial={(height, width)}"
        )
    return mask.to(device=device, dtype=torch.float32)


def _flatten_dynamic_weight(
    density_weight: Optional[torch.Tensor],
    layout: DynamicLayout,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    if density_weight is None:
        return None
    batch_size = layout.batch_size
    num_frames = layout.num_frames
    height = layout.height
    width = layout.width
    weight = torch.as_tensor(density_weight, device=device, dtype=dtype)

    if weight.ndim == 1:
        if weight.shape[0] != width:
            raise ValueError(f"Width-only density weight must have size {width}, got {tuple(weight.shape)}")
        return weight.view(1, 1, 1, width).expand(batch_size * num_frames, 1, 1, width)
    if weight.ndim == 2:
        if tuple(weight.shape) == (height, width):
            return weight.view(1, 1, height, width).expand(batch_size * num_frames, 1, height, width)
        if tuple(weight.shape) == (num_frames, width):
            return weight.view(1, num_frames, 1, width).expand(batch_size, num_frames, 1, width).reshape(
                batch_size * num_frames, 1, 1, width
            )
        if tuple(weight.shape) == (batch_size, width):
            return weight.view(batch_size, 1, 1, width).expand(batch_size, num_frames, 1, width).reshape(
                batch_size * num_frames, 1, 1, width
            )
        raise ValueError(f"Unsupported dynamic density-weight shape {tuple(weight.shape)}")
    if weight.ndim == 3:
        if tuple(weight.shape) == (num_frames, height, width):
            return weight.unsqueeze(0).expand(batch_size, num_frames, height, width).reshape(
                batch_size * num_frames, 1, height, width
            )
        if tuple(weight.shape) == (batch_size, height, width):
            return weight.unsqueeze(1).expand(batch_size, num_frames, height, width).reshape(
                batch_size * num_frames, 1, height, width
            )
        if tuple(weight.shape) == (batch_size, num_frames, width):
            return weight.reshape(batch_size * num_frames, 1, 1, width)
        raise ValueError(f"Unsupported dynamic density-weight shape {tuple(weight.shape)}")
    if weight.ndim == 4:
        if tuple(weight.shape) == (num_frames, 1, height, width):
            return weight.unsqueeze(0).expand(batch_size, num_frames, 1, height, width).reshape(
                batch_size * num_frames, 1, height, width
            )
        if tuple(weight.shape) == (batch_size, num_frames, height, width):
            return weight.reshape(batch_size * num_frames, 1, height, width)
        if tuple(weight.shape) == (batch_size, num_frames, 1, width):
            return weight.reshape(batch_size * num_frames, 1, 1, width)
        raise ValueError(f"Unsupported dynamic density-weight shape {tuple(weight.shape)}")
    if weight.ndim == 5 and tuple(weight.shape) == (batch_size, num_frames, 1, height, width):
        return weight.reshape(batch_size * num_frames, 1, height, width)
    raise ValueError(f"Unsupported dynamic density-weight shape {tuple(weight.shape)}")


def _prepare_dynamic_image_problem(
    image: torch.Tensor,
    maps: torch.Tensor,
    mask: torch.Tensor,
    density_weight: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], DynamicLayout]:
    image_5d, squeezed_batch = _ensure_dynamic_image(image)
    batch_size, num_frames, _, height, width = image_5d.shape
    maps_4d = _ensure_dynamic_maps(maps, batch_size, height, width, image_5d.device)
    num_coils = maps_4d.shape[1]
    mask_5d = _ensure_dynamic_mask(mask, batch_size, num_frames, height, width, image_5d.device)
    layout = DynamicLayout(batch_size, num_frames, num_coils, height, width, squeezed_batch)

    image_flat = image_5d.reshape(batch_size * num_frames, 1, height, width)
    maps_flat = maps_4d[:, None, ...].expand(batch_size, num_frames, num_coils, height, width).reshape(
        batch_size * num_frames,
        num_coils,
        height,
        width,
    )
    mask_flat = mask_5d.reshape(batch_size * num_frames, 1, height, width)
    weight_flat = _flatten_dynamic_weight(density_weight, layout, image_5d.device, image_5d.real.dtype)
    return image_flat, maps_flat, mask_flat, weight_flat, layout


def _prepare_dynamic_kspace_problem(
    kspace: torch.Tensor,
    maps: torch.Tensor,
    mask: torch.Tensor,
    density_weight: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], DynamicLayout]:
    kspace_5d, squeezed_batch = _ensure_dynamic_kspace(kspace)
    batch_size, num_frames, num_coils, height, width = kspace_5d.shape
    maps_4d = _ensure_dynamic_maps(maps, batch_size, height, width, kspace_5d.device)
    if maps_4d.shape[1] != num_coils:
        raise ValueError(f"Map coils={maps_4d.shape[1]} do not match kspace coils={num_coils}")
    mask_5d = _ensure_dynamic_mask(mask, batch_size, num_frames, height, width, kspace_5d.device)
    layout = DynamicLayout(batch_size, num_frames, num_coils, height, width, squeezed_batch)

    kspace_flat = kspace_5d.reshape(batch_size * num_frames, num_coils, height, width)
    maps_flat = maps_4d[:, None, ...].expand(batch_size, num_frames, num_coils, height, width).reshape(
        batch_size * num_frames,
        num_coils,
        height,
        width,
    )
    mask_flat = mask_5d.reshape(batch_size * num_frames, 1, height, width)
    weight_flat = _flatten_dynamic_weight(density_weight, layout, kspace_5d.device, kspace_5d.real.dtype)
    return kspace_flat, maps_flat, mask_flat, weight_flat, layout


def _restore_dynamic_image(image_flat: torch.Tensor, layout: DynamicLayout) -> torch.Tensor:
    image = image_flat.reshape(layout.batch_size, layout.num_frames, 1, layout.height, layout.width)
    return image[0] if layout.squeezed_batch else image


def _restore_dynamic_kspace(kspace_flat: torch.Tensor, layout: DynamicLayout) -> torch.Tensor:
    kspace = kspace_flat.reshape(
        layout.batch_size,
        layout.num_frames,
        layout.num_coils,
        layout.height,
        layout.width,
    )
    return kspace[0] if layout.squeezed_batch else kspace


def dynamic_a_forward(image: torch.Tensor, maps: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    image_flat, maps_flat, mask_flat, _, layout = _prepare_dynamic_image_problem(image, maps, mask)
    return _restore_dynamic_kspace(sense_forward(image_flat, maps_flat, mask_flat), layout)


def dynamic_a_adjoint(kspace: torch.Tensor, maps: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    kspace_flat, maps_flat, mask_flat, _, layout = _prepare_dynamic_kspace_problem(kspace, maps, mask)
    return _restore_dynamic_image(sense_adjoint(kspace_flat, maps_flat, mask_flat), layout)


def dynamic_a_normal(image: torch.Tensor, maps: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    image_flat, maps_flat, mask_flat, _, layout = _prepare_dynamic_image_problem(image, maps, mask)
    return _restore_dynamic_image(sense_normal(image_flat, maps_flat, mask_flat), layout)


def dynamic_weighted_forward(
    image: torch.Tensor,
    maps: torch.Tensor,
    mask: torch.Tensor,
    density_weight: Optional[torch.Tensor],
) -> torch.Tensor:
    image_flat, maps_flat, mask_flat, weight_flat, layout = _prepare_dynamic_image_problem(
        image,
        maps,
        mask,
        density_weight=density_weight,
    )
    out = sense_weighted_forward(image_flat, maps_flat, mask_flat, weight_flat)
    return _restore_dynamic_kspace(out, layout)


def dynamic_weighted_adjoint(
    kspace: torch.Tensor,
    maps: torch.Tensor,
    mask: torch.Tensor,
    density_weight: Optional[torch.Tensor],
) -> torch.Tensor:
    kspace_flat, maps_flat, mask_flat, weight_flat, layout = _prepare_dynamic_kspace_problem(
        kspace,
        maps,
        mask,
        density_weight=density_weight,
    )
    out = sense_weighted_adjoint(kspace_flat, maps_flat, mask_flat, weight_flat)
    return _restore_dynamic_image(out, layout)


def dynamic_weighted_normal(
    image: torch.Tensor,
    maps: torch.Tensor,
    mask: torch.Tensor,
    density_weight: Optional[torch.Tensor],
) -> torch.Tensor:
    image_flat, maps_flat, mask_flat, weight_flat, layout = _prepare_dynamic_image_problem(
        image,
        maps,
        mask,
        density_weight=density_weight,
    )
    out = sense_weighted_normal(image_flat, maps_flat, mask_flat, weight_flat)
    return _restore_dynamic_image(out, layout)


def full_sense_pinv(
    kspace: torch.Tensor,
    maps: torch.Tensor,
    l2lam: float = 0.0,
    eps: float = EPS,
) -> torch.Tensor:
    kspace_flat, maps_flat, _, _, layout = _prepare_dynamic_kspace_problem(
        kspace,
        maps,
        torch.ones(kspace.shape[:-3] + kspace.shape[-2:], device=kspace.device),
    )
    coil_imgs = ifft2c(kspace_flat)
    numerator = torch.sum(torch.conj(maps_flat) * coil_imgs, dim=1, keepdim=True)
    power = torch.sum(torch.abs(maps_flat) ** 2, dim=1, keepdim=True).clamp_min(float(eps))
    recon = numerator / (power + float(l2lam))
    return _restore_dynamic_image(recon, layout)

