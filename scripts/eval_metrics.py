from __future__ import annotations

from typing import Dict

import numpy as np
import torch

try:
    from skimage.metrics import structural_similarity
except ImportError:  # pragma: no cover - optional runtime dependency in some environments.
    structural_similarity = None


EPS = 1e-8


def to_magnitude(x: torch.Tensor) -> torch.Tensor:
    if torch.is_complex(x):
        return torch.abs(x)
    if x.ndim >= 1 and x.shape[-1] == 2:
        return torch.linalg.vector_norm(x, dim=-1)
    return x.abs() if x.is_floating_point() else x.float()


def ensure_bthw(x: torch.Tensor) -> torch.Tensor:
    mag = to_magnitude(x).float()
    if mag.ndim == 5 and mag.shape[2] == 1:
        return mag.squeeze(2)
    if mag.ndim == 4:
        return mag
    if mag.ndim == 3:
        return mag.unsqueeze(0)
    raise ValueError(f"Expected [B, T, 1, H, W], [B, T, H, W], [T, 1, H, W], or [T, H, W], got {tuple(mag.shape)}")


def nmse_per_sample(prediction: torch.Tensor, target: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    pred = ensure_bthw(prediction)
    ref = ensure_bthw(target)
    if pred.shape != ref.shape:
        raise ValueError(f"prediction shape {tuple(pred.shape)} does not match target shape {tuple(ref.shape)}")
    numerator = torch.sum((pred - ref) ** 2, dim=(1, 2, 3))
    denominator = torch.sum(ref ** 2, dim=(1, 2, 3)).clamp_min(float(eps))
    return numerator / denominator


def mean_nmse(prediction: torch.Tensor, target: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    return nmse_per_sample(prediction, target, eps=eps).mean()


def nrmse_per_sample(prediction: torch.Tensor, target: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    return torch.sqrt(nmse_per_sample(prediction, target, eps=eps))


def psnr_per_sample(prediction: torch.Tensor, target: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    pred = ensure_bthw(prediction)
    ref = ensure_bthw(target)
    mse = torch.mean((pred - ref) ** 2, dim=(1, 2, 3)).clamp_min(float(eps))
    max_pixel = torch.amax(ref, dim=(1, 2, 3)).clamp_min(float(eps))
    return 20.0 * torch.log10(max_pixel) - 10.0 * torch.log10(mse)


def frame_difference_nmse_per_sample(prediction: torch.Tensor, target: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    pred = ensure_bthw(prediction)
    ref = ensure_bthw(target)
    if pred.shape[1] <= 1:
        return torch.zeros(pred.shape[0], device=pred.device, dtype=pred.dtype)
    return nmse_per_sample(pred[:, 1:] - pred[:, :-1], ref[:, 1:] - ref[:, :-1], eps=eps)


def ssim_per_sample(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = ensure_bthw(prediction).detach().cpu().numpy()
    ref = ensure_bthw(target).detach().cpu().numpy()
    if structural_similarity is None:
        return torch.full((pred.shape[0],), float("nan"))

    values = []
    for batch_idx in range(pred.shape[0]):
        frame_scores = []
        for frame_idx in range(pred.shape[1]):
            gt_frame = ref[batch_idx, frame_idx]
            pred_frame = pred[batch_idx, frame_idx]
            data_range = float(np.max(gt_frame) - np.min(gt_frame))
            if data_range <= 0:
                data_range = 1.0
            frame_scores.append(float(structural_similarity(gt_frame, pred_frame, data_range=data_range)))
        values.append(float(np.mean(frame_scores)))
    return torch.tensor(values, dtype=torch.float32)


def summarize_reconstruction_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    include_ssim: bool = False,
) -> Dict[str, float]:
    metrics = {
        "nmse": float(mean_nmse(prediction, target)),
        "nrmse": float(nrmse_per_sample(prediction, target).mean()),
        "psnr": float(psnr_per_sample(prediction, target).mean()),
        "frame_diff_nmse": float(frame_difference_nmse_per_sample(prediction, target).mean()),
    }
    if include_ssim:
        metrics["ssim"] = float(ssim_per_sample(prediction, target).mean())
    return metrics
