from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cardiac_ensure.scripts.train_supervised_baseline import run_experiment as run_supervised
from cardiac_ensure.scripts.train_true_ensure import run_experiment as run_true_ensure


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--preproc-root", type=Path, required=True)
    parser.add_argument("--density-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--window-size", type=int, default=5)
    return parser


def _device_string(device: str | None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    device = _device_string(args.device)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    common_kwargs = {
        "data_root": args.data_root,
        "preproc_root": args.preproc_root,
        "density_root": args.density_root,
        "train_split": "train",
        "val_split": "val",
        "acceleration": 4.0,
        "sigma_mask": 0.18,
        "window_size": args.window_size,
        "stride": 1,
        "window_mode": "centered",
        "center_slice_fraction": 1.0,
        "epochs": 1,
        "batch_size": 1,
        "num_workers": 0,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "grad_clip": 1.0,
        "chans": 64,
        "num_pools": 4,
        "num_unrolls": 3,
        "drop_prob": 0.0,
        "no_residual": False,
        "device": device,
        "seed": args.seed,
        "save_every": 1,
        "include_ssim": False,
        "train_deterministic_masks": True,
        "val_deterministic_masks": True,
        "max_train_steps": 1,
        "max_val_batches": 1,
        "crop_height": 64,
        "crop_width": 96,
    }

    supervised_args = argparse.Namespace(
        **common_kwargs,
        output_dir=output_dir / "supervised",
        frame_mode="all",
    )
    true_ensure_args = argparse.Namespace(
        **common_kwargs,
        output_dir=output_dir / "true_ensure",
        cg_l2lam=1e-6,
        cg_max_iter=5,
        cg_tol=1e-6,
        divergence_eps=None,
        divergence_mc_samples=1,
        compute_val_risk=True,
    )

    supervised_summary = run_supervised(supervised_args)
    true_ensure_summary = run_true_ensure(true_ensure_args)

    supervised_last = supervised_summary["history"][-1]
    true_ensure_last = true_ensure_summary["history"][-1]

    acceptance = {
        "supervised_forward_backward": torch.isfinite(torch.tensor(supervised_last["train_loss"])).item(),
        "supervised_val_nmse_finite": torch.isfinite(torch.tensor(supervised_last["val_nmse"])).item(),
        "true_ensure_forward_backward": torch.isfinite(torch.tensor(true_ensure_last["train_loss"])).item(),
        "true_ensure_val_nmse_finite": torch.isfinite(torch.tensor(true_ensure_last["val_nmse"])).item(),
        "true_ensure_gt_free_train_step": not bool(true_ensure_last["observed_target_in_train"]),
    }

    summary = {
        "device": device,
        "supervised": supervised_summary,
        "true_ensure": true_ensure_summary,
        "acceptance": acceptance,
    }
    with (output_dir / "smoke_summary.json").open("w", encoding="utf-8") as fobj:
        json.dump(summary, fobj, indent=2)

    print(json.dumps(summary, indent=2))
    if not all(acceptance.values()):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
