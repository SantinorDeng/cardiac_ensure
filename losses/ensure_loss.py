from __future__ import annotations

from typing import Callable, Dict, Optional

import torch

from cardiac_ensure.ops.cg_solver import solve_rho_ls, solve_weighted_projection


def _real_inner(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.ndim == 4:
        return torch.real(torch.sum(torch.conj(x) * y)).reshape(1)
    return torch.real(torch.sum(torch.conj(x) * y, dim=tuple(range(1, x.ndim))))


def projected_energy(projected_error: torch.Tensor) -> torch.Tensor:
    if projected_error.ndim == 4:
        return torch.real(torch.sum(torch.conj(projected_error) * projected_error, dim=(-3, -2, -1)))
    return torch.real(torch.sum(torch.conj(projected_error) * projected_error, dim=(-3, -2, -1)))


def ensure_data_term(
    prediction: torch.Tensor,
    rho_ls: torch.Tensor,
    maps: torch.Tensor,
    mask: torch.Tensor,
    density_weight: Optional[torch.Tensor],
    l2lam: float = 1e-6,
    max_iter: int = 25,
    tol: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    error = prediction - rho_ls
    projected, projection_info = solve_weighted_projection(
        error=error,
        maps=maps,
        mask=mask,
        density_weight=density_weight,
        l2lam=l2lam,
        max_iter=max_iter,
        tol=tol,
    )
    frame_energy = projected_energy(projected)
    return {
        "projected_error": projected,
        "frame_energy": frame_energy,
        "data_term": frame_energy.mean(),
        "projection_info": projection_info,
    }


def estimate_divergence_mc(
    model_fn: Callable[[torch.Tensor], torch.Tensor],
    inputs: torch.Tensor,
    eps: Optional[float] = None,
    num_mc_samples: int = 1,
    post_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    if num_mc_samples <= 0:
        raise ValueError(f"num_mc_samples must be positive, got {num_mc_samples}")
    if not torch.is_complex(inputs):
        raise TypeError(f"inputs must be complex, got {inputs.dtype}")

    if eps is None:
        rms = torch.sqrt(torch.mean(torch.abs(inputs) ** 2)).detach().item()
        eps = max(1e-6, 1e-3 * float(rms))

    base_output = model_fn(inputs)
    if post_fn is not None:
        base_output = post_fn(base_output)

    estimates = []
    for _ in range(int(num_mc_samples)):
        with torch.no_grad():
            noise = torch.complex(torch.randn_like(inputs.real), torch.randn_like(inputs.real))
        perturbed_output = model_fn(inputs + float(eps) * noise)
        if post_fn is not None:
            perturbed_output = post_fn(perturbed_output)
        estimates.append(_real_inner(noise, perturbed_output - base_output) / float(eps))

    per_sample = torch.stack(estimates, dim=0).mean(dim=0)
    return {
        "divergence_per_sample": per_sample,
        "divergence": per_sample.mean(),
        "eps": torch.tensor(float(eps), device=inputs.device, dtype=inputs.real.dtype),
    }


def compute_true_ensure_loss(
    model_fn: Callable[[torch.Tensor], torch.Tensor],
    zf_input: torch.Tensor,
    kspace: torch.Tensor,
    maps: torch.Tensor,
    mask: torch.Tensor,
    noise_sigma2: torch.Tensor,
    density_weight: Optional[torch.Tensor],
    cg_l2lam: float = 1e-6,
    cg_max_iter: int = 25,
    cg_tol: float = 1e-6,
    divergence_eps: Optional[float] = None,
    divergence_mc_samples: int = 1,
    divergence_post_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    prediction = model_fn(zf_input)
    rho_ls, rho_info = solve_rho_ls(
        kspace=kspace,
        maps=maps,
        mask=mask,
        l2lam=cg_l2lam,
        max_iter=cg_max_iter,
        tol=cg_tol,
    )
    data_out = ensure_data_term(
        prediction=prediction,
        rho_ls=rho_ls,
        maps=maps,
        mask=mask,
        density_weight=density_weight,
        l2lam=cg_l2lam,
        max_iter=cg_max_iter,
        tol=cg_tol,
    )
    div_out = estimate_divergence_mc(
        model_fn=model_fn,
        inputs=zf_input,
        eps=divergence_eps,
        num_mc_samples=divergence_mc_samples,
        post_fn=divergence_post_fn,
    )
    sigma2 = torch.as_tensor(noise_sigma2, device=zf_input.device, dtype=zf_input.real.dtype).mean()
    div_scale = 2.0 * sigma2
    div_contribution = div_scale * div_out["divergence"]
    loss = data_out["data_term"] + div_contribution
    return {
        "loss": loss,
        "data_term": data_out["data_term"],
        "div_term": div_out["divergence"],
        "div_scale": div_scale,
        "div_contribution": div_contribution,
        "risk_proxy": loss,
        "prediction": prediction,
        "rho_ls": rho_ls,
        "projected_error": data_out["projected_error"],
        "rho_info": rho_info,
        "projection_info": data_out["projection_info"],
        "frame_energy": data_out["frame_energy"],
        "divergence_per_sample": div_out["divergence_per_sample"],
        "divergence_eps": div_out["eps"],
    }
