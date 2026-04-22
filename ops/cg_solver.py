from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch

from .mri_ops import (
    _prepare_dynamic_image_problem,
    _prepare_dynamic_kspace_problem,
    _restore_dynamic_image,
    sense_adjoint,
    sense_normal,
    sense_weighted_adjoint,
    sense_weighted_forward,
    sense_weighted_normal,
)


EPS = 1e-12


@dataclass
class CGInfo:
    iterations: int
    converged: torch.Tensor
    residual_norms: torch.Tensor

    def to_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "iterations": torch.tensor(self.iterations, dtype=torch.int32),
            "converged": self.converged,
            "residual_norms": self.residual_norms,
        }


def _complex_sqnorm(x: torch.Tensor) -> torch.Tensor:
    return torch.real(torch.sum(torch.conj(x) * x, dim=tuple(range(1, x.ndim))))


def complex_conjugate_gradient(
    operator: Callable[[torch.Tensor], torch.Tensor],
    rhs: torch.Tensor,
    x0: Optional[torch.Tensor] = None,
    max_iter: int = 25,
    tol: float = 1e-6,
) -> Tuple[torch.Tensor, CGInfo]:
    if rhs.ndim < 2:
        raise ValueError(f"rhs must have a batch dimension and at least one feature dimension, got {tuple(rhs.shape)}")

    x = torch.zeros_like(rhs) if x0 is None else x0.clone()
    r = rhs - operator(x)
    p = r.clone()
    rsold = _complex_sqnorm(r)
    history = [torch.sqrt(rsold.clamp_min(0.0))]

    num_iter = 0
    for idx in range(int(max_iter)):
        if bool(torch.all(rsold <= float(tol) ** 2)):
            break
        Ap = operator(p)
        denom = torch.real(torch.sum(torch.conj(p) * Ap, dim=tuple(range(1, p.ndim)))).clamp_min(EPS)
        alpha = rsold / denom
        alpha_view = alpha.reshape(-1, *([1] * (rhs.ndim - 1)))

        x = x + alpha_view * p
        r = r - alpha_view * Ap
        rsnew = _complex_sqnorm(r)
        history.append(torch.sqrt(rsnew.clamp_min(0.0)))
        num_iter = idx + 1

        if bool(torch.all(rsnew <= float(tol) ** 2)):
            rsold = rsnew
            break

        beta = rsnew / rsold.clamp_min(EPS)
        beta_view = beta.reshape(-1, *([1] * (rhs.ndim - 1)))
        p = r + beta_view * p
        rsold = rsnew

    residual_norms = torch.stack(history, dim=0)
    converged = residual_norms[-1] <= float(tol)
    return x, CGInfo(iterations=num_iter, converged=converged, residual_norms=residual_norms)


def solve_rho_ls(
    kspace: torch.Tensor,
    maps: torch.Tensor,
    mask: torch.Tensor,
    l2lam: float = 1e-6,
    max_iter: int = 25,
    tol: float = 1e-6,
    x0: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, CGInfo]:
    kspace_flat, maps_flat, mask_flat, _, layout = _prepare_dynamic_kspace_problem(kspace, maps, mask)
    rhs = sense_adjoint(kspace_flat, maps_flat, mask_flat)
    if x0 is None:
        x0_flat = rhs
    else:
        x0_flat, _, _, _, _ = _prepare_dynamic_image_problem(x0, maps, mask)

    operator = lambda image: sense_normal(image, maps_flat, mask_flat) + float(l2lam) * image
    rho_ls_flat, info = complex_conjugate_gradient(
        operator=operator,
        rhs=rhs,
        x0=x0_flat,
        max_iter=max_iter,
        tol=tol,
    )
    return _restore_dynamic_image(rho_ls_flat, layout), info


def solve_weighted_projection(
    error: torch.Tensor,
    maps: torch.Tensor,
    mask: torch.Tensor,
    density_weight: Optional[torch.Tensor],
    l2lam: float = 1e-6,
    max_iter: int = 25,
    tol: float = 1e-6,
    x0: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, CGInfo]:
    error_flat, maps_flat, mask_flat, weight_flat, layout = _prepare_dynamic_image_problem(
        error,
        maps,
        mask,
        density_weight=density_weight,
    )
    rhs = sense_weighted_adjoint(
        sense_weighted_forward(error_flat, maps_flat, mask_flat, weight_flat),
        maps_flat,
        mask_flat,
        weight_flat,
    )
    x0_flat = torch.zeros_like(error_flat) if x0 is None else _prepare_dynamic_image_problem(
        x0,
        maps,
        mask,
        density_weight=density_weight,
    )[0]
    operator = lambda image: sense_weighted_normal(image, maps_flat, mask_flat, weight_flat) + float(l2lam) * image
    projection_flat, info = complex_conjugate_gradient(
        operator=operator,
        rhs=rhs,
        x0=x0_flat,
        max_iter=max_iter,
        tol=tol,
    )
    return _restore_dynamic_image(projection_flat, layout), info
