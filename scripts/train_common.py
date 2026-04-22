from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping

import numpy as np
import torch


def configure_torch() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_to_device(batch: Any, device: torch.device) -> Any:
    if torch.is_tensor(batch):
        return batch.to(device)
    if isinstance(batch, dict):
        return {key: move_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, list):
        return [move_to_device(value, device) for value in batch]
    if isinstance(batch, tuple):
        return tuple(move_to_device(value, device) for value in batch)
    return batch


def _crop_last_dim(x: torch.Tensor, target: int | None) -> torch.Tensor:
    if target is None or x.shape[-1] <= target:
        return x
    start = (x.shape[-1] - target) // 2
    return x[..., start : start + target]


def _crop_last_two_dims(
    x: torch.Tensor,
    crop_height: int | None,
    crop_width: int | None,
) -> torch.Tensor:
    if crop_height is not None and x.shape[-2] > crop_height:
        start = (x.shape[-2] - crop_height) // 2
        x = x[..., start : start + crop_height, :]
    if crop_width is not None and x.shape[-1] > crop_width:
        start = (x.shape[-1] - crop_width) // 2
        x = x[..., start : start + crop_width]
    return x


def maybe_center_crop_batch(
    batch: Mapping[str, Any],
    crop_height: int | None = None,
    crop_width: int | None = None,
) -> Dict[str, Any]:
    if crop_height is None and crop_width is None:
        return dict(batch)

    out: Dict[str, Any] = {}
    spatial_keys = {"kspace_fs", "kspace_us", "mask", "mask_prob", "zf", "target_rss", "maps"}
    width_only_keys = {"empirical_density", "inv_sqrt_density"}
    for key, value in batch.items():
        if torch.is_tensor(value):
            if key in spatial_keys:
                out[key] = _crop_last_two_dims(value, crop_height=crop_height, crop_width=crop_width)
            elif key in width_only_keys:
                out[key] = _crop_last_dim(value, crop_width)
            else:
                out[key] = value
        else:
            out[key] = value
    return out


def select_frame_mode(x: torch.Tensor, frame_mode: str) -> torch.Tensor:
    if frame_mode == "all":
        return x
    if frame_mode != "center":
        raise ValueError(f"Unsupported frame_mode={frame_mode}")
    temporal_dim = 1 if x.ndim >= 2 else 0
    center = x.shape[temporal_dim] // 2
    index = [slice(None)] * x.ndim
    index[temporal_dim] = slice(center, center + 1)
    return x[tuple(index)]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def serialize_for_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, dict):
        return {key: serialize_for_json(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_for_json(item) for item in value]
    return value


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as fobj:
        json.dump(serialize_for_json(dict(payload)), fobj, indent=2, sort_keys=True)


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters() if param.requires_grad))


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: list[dict[str, Any]],
    config: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> None:
    payload: Dict[str, Any] = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
        "config": serialize_for_json(config),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


@dataclass
class RunningStats:
    totals: MutableMapping[str, float] = field(default_factory=dict)
    count: int = 0

    def update(self, values: Mapping[str, float]) -> None:
        self.count += 1
        for key, value in values.items():
            self.totals[key] = self.totals.get(key, 0.0) + float(value)

    def averages(self) -> Dict[str, float]:
        if self.count == 0:
            return {key: float("nan") for key in self.totals}
        return {key: value / float(self.count) for key, value in self.totals.items()}
