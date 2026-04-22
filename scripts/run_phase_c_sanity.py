from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cardiac_ensure.datasets import CardiacCineENSUREDataset
from cardiac_ensure.losses.ensure_loss import estimate_divergence_mc
from cardiac_ensure.ops.cg_solver import solve_rho_ls, solve_weighted_projection
from cardiac_ensure.ops.mri_ops import (
    dynamic_a_adjoint,
    dynamic_a_forward,
    dynamic_a_normal,
    full_sense_pinv,
)


def _complex_randn_like(x: torch.Tensor) -> torch.Tensor:
    return torch.complex(torch.randn_like(x.real), torch.randn_like(x.real))


def _relative_error(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> float:
    return float(torch.linalg.norm((x - y).reshape(-1)) / (torch.linalg.norm(y.reshape(-1)) + eps))


def _history_is_monotone(residual_norms: torch.Tensor, atol: float = 1e-7) -> bool:
    diffs = residual_norms[1:] - residual_norms[:-1]
    return bool(torch.all(diffs <= float(atol)))


def _temporal_surrogate(zf: torch.Tensor) -> torch.Tensor:
    return 0.25 * torch.roll(zf, shifts=1, dims=0) + 0.5 * zf + 0.25 * torch.roll(zf, shifts=-1, dims=0)


def _high_frequency_probe(reference: torch.Tensor, scale: float = 0.1) -> torch.Tensor:
    _, _, height, width = reference.shape
    checker = ((-1.0) ** (torch.arange(height)[:, None] + torch.arange(width)[None, :])).to(reference.real.dtype)
    checker = checker.to(reference.device)[None, None, ...].expand(reference.shape[0], 1, height, width)
    amplitude = float(scale) * torch.mean(torch.abs(reference)).detach()
    return amplitude * checker.to(reference.dtype)


def _load_sample(args: argparse.Namespace) -> Dict[str, torch.Tensor]:
    dataset = CardiacCineENSUREDataset(
        root=args.data_root,
        split=args.split,
        preproc_root=args.preproc_root,
        density_root=args.density_root,
        acceleration=args.acceleration,
        sigma_mask=args.sigma_mask,
        window_size=args.window_size,
        stride=args.stride,
        center_slice_fraction=args.center_slice_fraction,
        deterministic_masks=True,
        mask_seed=args.seed,
    )
    sample = dataset[args.sample_index]
    out: Dict[str, torch.Tensor] = {}
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(args.device)
        else:
            out[key] = value
    return out


def run_c1(sample: Dict[str, torch.Tensor]) -> Dict[str, float]:
    maps = sample["maps"]
    mask = sample["mask"]
    x_ref = full_sense_pinv(sample["kspace_fs"], maps, l2lam=1e-6)
    y_probe = _complex_randn_like(sample["kspace_us"])

    ax = dynamic_a_forward(x_ref, maps, mask)
    aha_x = dynamic_a_normal(x_ref, maps, mask)
    lhs = torch.sum(torch.conj(ax) * y_probe)
    rhs = torch.sum(torch.conj(x_ref) * dynamic_a_adjoint(y_probe, maps, mask))
    adjoint_rel_err = float(torch.abs(lhs - rhs) / torch.clamp(torch.abs(lhs), min=1e-8))

    single_ax = dynamic_a_forward(x_ref[:1], maps, mask[:1])
    single_aha = dynamic_a_normal(x_ref[:1], maps, mask[:1])

    return {
        "adjoint_relative_error": adjoint_rel_err,
        "ax_shape_match": float(tuple(ax.shape) == tuple(sample["kspace_us"].shape)),
        "aha_shape_match": float(tuple(aha_x.shape) == tuple(x_ref.shape)),
        "single_frame_kspace_shape_match": float(tuple(single_ax.shape) == (1, maps.shape[0], maps.shape[-2], maps.shape[-1])),
        "single_frame_image_shape_match": float(tuple(single_aha.shape) == (1, 1, maps.shape[-2], maps.shape[-1])),
        "aha_over_x_norm": float(torch.linalg.norm(aha_x.reshape(-1)) / torch.clamp(torch.linalg.norm(x_ref.reshape(-1)), min=1e-8)),
        "all_finite": float(torch.isfinite(ax.real).all() and torch.isfinite(aha_x.real).all()),
    }


def run_c2(sample: Dict[str, torch.Tensor], args: argparse.Namespace) -> Dict[str, float]:
    maps = sample["maps"]
    mask = sample["mask"]
    kspace_us = sample["kspace_us"]
    kspace_fs = sample["kspace_fs"]
    full_mask = torch.ones_like(mask)

    rho_ls_us, info_us = solve_rho_ls(
        kspace=kspace_us,
        maps=maps,
        mask=mask,
        l2lam=args.cg_l2lam,
        max_iter=args.cg_max_iter,
        tol=args.cg_tol,
    )
    rho_ls_fs, info_fs = solve_rho_ls(
        kspace=kspace_fs,
        maps=maps,
        mask=full_mask,
        l2lam=args.cg_l2lam,
        max_iter=args.cg_ref_max_iter,
        tol=args.cg_ref_tol,
    )
    rho_ref = full_sense_pinv(kspace_fs, maps, l2lam=args.cg_l2lam)

    return {
        "undersampled_residual_monotone": float(_history_is_monotone(info_us.residual_norms)),
        "fullysampled_residual_monotone": float(_history_is_monotone(info_fs.residual_norms)),
        "fullysampled_relative_error": _relative_error(rho_ls_fs, rho_ref),
        "undersampled_final_residual": float(info_us.residual_norms[-1].mean()),
        "fullysampled_final_residual": float(info_fs.residual_norms[-1].mean()),
        "rho_ls_finite": float(torch.isfinite(rho_ls_us.real).all()),
    }


def run_c3(sample: Dict[str, torch.Tensor], args: argparse.Namespace) -> Dict[str, float]:
    maps = sample["maps"]
    mask = sample["mask"]
    rho_ls, _ = solve_rho_ls(
        kspace=sample["kspace_us"],
        maps=maps,
        mask=mask,
        l2lam=args.cg_l2lam,
        max_iter=args.cg_max_iter,
        tol=args.cg_tol,
    )
    error = _high_frequency_probe(rho_ls)
    weighted, weighted_info = solve_weighted_projection(
        error=error,
        maps=maps,
        mask=mask,
        density_weight=sample.get("inv_sqrt_density"),
        l2lam=args.cg_l2lam,
        max_iter=args.cg_max_iter,
        tol=args.cg_tol,
        x0=torch.zeros_like(error),
    )
    unweighted, _ = solve_weighted_projection(
        error=error,
        maps=maps,
        mask=mask,
        density_weight=None,
        l2lam=args.cg_l2lam,
        max_iter=args.cg_max_iter,
        tol=args.cg_tol,
        x0=torch.zeros_like(error),
    )
    zero_proj, _ = solve_weighted_projection(
        error=torch.zeros_like(error),
        maps=maps,
        mask=mask,
        density_weight=sample.get("inv_sqrt_density"),
        l2lam=args.cg_l2lam,
        max_iter=args.cg_max_iter,
        tol=args.cg_tol,
        x0=torch.zeros_like(error),
    )

    diff = torch.linalg.norm((weighted - unweighted).reshape(-1))
    denom = torch.clamp(torch.linalg.norm(unweighted.reshape(-1)), min=1e-8)
    return {
        "projection_initial_residual": float(weighted_info.residual_norms[0].mean()),
        "projection_final_residual": float(weighted_info.residual_norms[-1].mean()),
        "weighted_vs_unweighted_relative_change": float(diff / denom),
        "zero_error_projection_norm": float(torch.linalg.norm(zero_proj.reshape(-1))),
        "projection_finite": float(torch.isfinite(weighted.real).all()),
    }


def run_c4(sample: Dict[str, torch.Tensor]) -> Dict[str, float]:
    alpha = 0.35
    linear_fn = lambda x: alpha * x
    v = sample["zf"]
    num_complex = v.numel()
    analytic = 2.0 * alpha * float(num_complex)
    eps_a = None
    eps_b = 1e-2

    div_a = estimate_divergence_mc(
        model_fn=linear_fn,
        inputs=v,
        eps=eps_a,
        num_mc_samples=8,
    )
    div_b = estimate_divergence_mc(
        model_fn=linear_fn,
        inputs=v,
        eps=eps_b,
        num_mc_samples=8,
    )

    est_a = float(div_a["divergence"])
    est_b = float(div_b["divergence"])
    return {
        "analytic_trace": analytic,
        "divergence_estimate_auto_eps": est_a,
        "divergence_estimate_eps_1e-2": est_b,
        "auto_eps_relative_error": abs(est_a - analytic) / max(abs(analytic), 1e-8),
        "large_eps_relative_error": abs(est_b - analytic) / max(abs(analytic), 1e-8),
        "divergence_finite": float(torch.isfinite(div_a["divergence"]) and torch.isfinite(div_b["divergence"])),
        "auto_eps_value": float(div_a["eps"]),
    }


def evaluate_acceptance(results: Dict[str, Dict[str, float]]) -> Dict[str, bool]:
    return {
        "C1": (
            results["C1"]["adjoint_relative_error"] < 1e-4
            and results["C1"]["ax_shape_match"] == 1.0
            and results["C1"]["aha_shape_match"] == 1.0
            and results["C1"]["single_frame_kspace_shape_match"] == 1.0
            and results["C1"]["single_frame_image_shape_match"] == 1.0
            and results["C1"]["all_finite"] == 1.0
        ),
        "C2": (
            results["C2"]["undersampled_residual_monotone"] == 1.0
            and results["C2"]["fullysampled_residual_monotone"] == 1.0
            and results["C2"]["fullysampled_relative_error"] < 5e-4
            and results["C2"]["rho_ls_finite"] == 1.0
        ),
        "C3": (
            results["C3"]["projection_finite"] == 1.0
            and results["C3"]["zero_error_projection_norm"] < 1e-8
            and results["C3"]["projection_final_residual"] < results["C3"]["projection_initial_residual"]
            and results["C3"]["weighted_vs_unweighted_relative_change"] > 5e-2
        ),
        "C4": (
            results["C4"]["divergence_finite"] == 1.0
            and results["C4"]["auto_eps_relative_error"] < 5e-2
            and results["C4"]["large_eps_relative_error"] < 1e-1
        ),
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--preproc-root", type=Path, default=None)
    parser.add_argument("--density-root", type=Path, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--center-slice-fraction", type=float, default=0.6)
    parser.add_argument("--acceleration", type=float, default=4.0)
    parser.add_argument("--sigma-mask", type=float, default=0.18)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--cg-l2lam", type=float, default=1e-6)
    parser.add_argument("--cg-max-iter", type=int, default=20)
    parser.add_argument("--cg-tol", type=float, default=1e-8)
    parser.add_argument("--cg-ref-max-iter", type=int, default=40)
    parser.add_argument("--cg-ref-tol", type=float, default=1e-10)
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    sample = _load_sample(args)
    results = {
        "C1": run_c1(sample),
        "C2": run_c2(sample, args),
        "C3": run_c3(sample, args),
        "C4": run_c4(sample),
    }
    acceptance = evaluate_acceptance(results)
    payload = {
        "sample_meta": sample["meta"],
        "results": results,
        "acceptance": acceptance,
        "all_passed": all(acceptance.values()),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
