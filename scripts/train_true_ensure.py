from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cardiac_ensure.datasets import CardiacCineENSUREDataset
from cardiac_ensure.losses import compute_true_ensure_loss
from cardiac_ensure.models import TemporalNormUnet
from cardiac_ensure.scripts.eval_metrics import summarize_reconstruction_metrics
from cardiac_ensure.scripts.train_common import (
    RunningStats,
    configure_torch,
    count_trainable_parameters,
    ensure_dir,
    maybe_center_crop_batch,
    move_to_device,
    resolve_device,
    save_checkpoint,
    save_json,
    set_random_seed,
)
"""
python /home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/scripts/train_true_ensure.py \
  --data-root /home/dengyipin/CMR2025/cmr001_pilot \
  --preproc-root /home/dengyipin/CMR2025/cmr001_pilot/preproc_c \
  --density-root /home/dengyipin/CMR2025/cmr001_pilot/density_stats \
  --train-split train \
  --val-split val \
  --output-dir /home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/outputs/pilot_true_ensure_r4_w5_unroll3 \
  --acceleration 4.0 \
  --sigma-mask 0.18 \
  --window-size 5 \
  --stride 1 \
  --window-mode centered \
  --center-slice-fraction 1.0 \
  --epochs 80 \
  --batch-size 1 \
  --num-workers 4 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --grad-clip 1.0 \
  --chans 64 \
  --num-pools 4 \
  --num-unrolls 3 \
  --drop-prob 0.0 \
  --device cuda:0 \
  --seed 7 \
  --save-every 1 \
  --val-deterministic-masks \
  --cg-l2lam 1e-6 \
  --cg-max-iter 25 \
  --cg-tol 1e-6 \
  --divergence-mc-samples 1 \
  --compute-val-risk

python /home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/scripts/train_true_ensure.py \
  --data-root /home/dengyipin/CMR2025/cmr001 \
  --preproc-root /home/dengyipin/CMR2025/cmr001/preproc_c \
  --density-root /home/dengyipin/CMR2025/cmr001/density_stats \
  --train-split train \
  --val-split val \
  --output-dir /home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/outputs/true_ensure_r4_w5_unroll3 \
  --acceleration 4.0 \
  --sigma-mask 0.18 \
  --window-size 5 \
  --stride 1 \
  --window-mode centered \
  --center-slice-fraction 1.0 \
  --epochs 25 \
  --batch-size 1 \
  --num-workers 4 \
  --lr 1e-6 \
  --weight-decay 1e-4 \
  --grad-clip 0.1 \
  --chans 64 \
  --num-pools 4 \
  --num-unrolls 3 \
  --drop-prob 0.0 \
  --device cuda:0 \
  --seed 7 \
  --save-every 1 \
  --train-deterministic-masks \
  --val-deterministic-masks \
  --cg-l2lam 1e-6 \
  --cg-max-iter 25 \
  --cg-tol 1e-6 \
  --divergence-mc-samples 2 \
  --compute-val-risk
"""

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--preproc-root", type=Path, default=None)
    parser.add_argument("--density-root", type=Path, default=None)
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="val")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--acceleration", type=float, default=4.0)
    parser.add_argument("--sigma-mask", type=float, default=0.18)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--window-mode", choices=("centered", "sliding"), default="centered")
    parser.add_argument("--center-slice-fraction", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--chans", type=int, default=64)
    parser.add_argument("--num-pools", type=int, default=4)
    parser.add_argument("--num-unrolls", type=int, default=3)
    parser.add_argument("--drop-prob", type=float, default=0.0)
    parser.add_argument("--no-residual", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--include-ssim", action="store_true")
    parser.add_argument("--train-deterministic-masks", action="store_true")
    parser.add_argument("--val-deterministic-masks", action="store_true")
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--crop-height", type=int, default=None)
    parser.add_argument("--crop-width", type=int, default=None)
    parser.add_argument("--cg-l2lam", type=float, default=1e-6)
    parser.add_argument("--cg-max-iter", type=int, default=25)
    parser.add_argument("--cg-tol", type=float, default=1e-6)
    parser.add_argument("--divergence-eps", type=float, default=None)
    parser.add_argument("--divergence-mc-samples", type=int, default=1)
    parser.add_argument("--divergence-mode", choices=("measurement", "image"), default="measurement")
    parser.add_argument("--no-divergence-projection", action="store_true")
    parser.add_argument("--save-step-debug", action="store_true")
    parser.add_argument("--step-debug-limit", type=int, default=None)
    parser.add_argument("--compute-val-risk", action="store_true")
    return parser


def _extract_step_diagnostics(
    model: TemporalNormUnet,
    loss_dict: Dict[str, torch.Tensor],
    step_idx: int,
) -> Dict[str, float | int | list[float]]:
    div_per_sample = loss_dict.get("divergence_per_sample")
    if div_per_sample is not None:
        div_per_sample_real = torch.as_tensor(div_per_sample).detach().real.reshape(-1)
        div_min = float(div_per_sample_real.min())
        div_max = float(div_per_sample_real.max())
        div_mean = float(div_per_sample_real.mean())
    else:
        div_min = float("nan")
        div_max = float("nan")
        div_mean = float("nan")

    dc_weights = [
        float(block.dc_weight_value().detach().cpu())
        for block in getattr(model, "dc_blocks", [])
        if hasattr(block, "dc_weight_value")
    ]
    diagnostics: Dict[str, float | int | list[float]] = {
        "step": int(step_idx),
        "loss": float(loss_dict["loss"].detach()),
        "data_term": float(loss_dict["data_term"].detach()),
        "div_term": float(loss_dict["div_term"].detach()),
        "div_scale": float(loss_dict["div_scale"].detach()),
        "div_contribution": float(loss_dict["div_contribution"].detach()),
        "divergence_eps": float(loss_dict["divergence_eps"].detach()),
        "divergence_per_sample_min": div_min,
        "divergence_per_sample_max": div_max,
        "divergence_per_sample_mean": div_mean,
        "dc_weights": dc_weights,
    }
    if dc_weights:
        diagnostics["dc_weight_min"] = min(dc_weights)
        diagnostics["dc_weight_max"] = max(dc_weights)
    return diagnostics


def _make_dataset(
    args: argparse.Namespace,
    split: str,
    deterministic_masks: bool,
    return_target: bool,
) -> CardiacCineENSUREDataset:
    return CardiacCineENSUREDataset(
        root=args.data_root,
        split=split,
        preproc_root=args.preproc_root,
        density_root=args.density_root,
        acceleration=args.acceleration,
        sigma_mask=args.sigma_mask,
        window_size=args.window_size,
        stride=args.stride,
        window_mode=args.window_mode,
        center_slice_fraction=args.center_slice_fraction,
        deterministic_masks=deterministic_masks,
        mask_seed=args.seed,
        return_target=return_target,
    )


def _make_loader(
    dataset: CardiacCineENSUREDataset,
    args: argparse.Namespace,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def _compute_train_loss(
    model: TemporalNormUnet,
    batch: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, torch.Tensor]:
    def model_fn(zf_input: torch.Tensor) -> torch.Tensor:
        return model(zf_input, maps=batch.get("maps"), mask=batch.get("mask"))

    return compute_true_ensure_loss(
        model_fn=model_fn,
        zf_input=batch["zf"],
        kspace=batch["kspace_us"],
        maps=batch["maps"],
        mask=batch["mask"],
        noise_sigma2=batch["noise_sigma2"],
        density_weight=batch.get("inv_sqrt_density"),
        cg_l2lam=args.cg_l2lam,
        cg_max_iter=args.cg_max_iter,
        cg_tol=args.cg_tol,
        divergence_eps=args.divergence_eps,
        divergence_mc_samples=args.divergence_mc_samples,
        divergence_mode=getattr(args, "divergence_mode", "measurement"),
        project_divergence=not getattr(args, "no_divergence_projection", False),
    )


def train_one_epoch(
    model: TemporalNormUnet,
    loader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, float]:
    model.train()
    stats = RunningStats()
    observed_target_in_train = False
    step_diagnostics: list[Dict[str, Any]] = []
    div_term_min: float | None = None
    div_term_max: float | None = None
    div_eps_min: float | None = None
    div_eps_max: float | None = None
    dc_weight_min: float | None = None
    dc_weight_max: float | None = None
    negative_div_steps = 0

    progress = tqdm(loader, desc="train", leave=False)
    for step_idx, batch in enumerate(progress):
        if args.max_train_steps is not None and step_idx >= args.max_train_steps:
            break

        batch = move_to_device(batch, device)
        batch = maybe_center_crop_batch(batch, crop_height=args.crop_height, crop_width=args.crop_width)
        observed_target_in_train = observed_target_in_train or ("target_rss" in batch)

        loss_dict = _compute_train_loss(model, batch, args)
        loss = loss_dict["loss"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        loss_value = float(loss.detach())
        data_term_value = float(loss_dict["data_term"].detach())
        div_term_value = float(loss_dict["div_term"].detach())
        div_scale_value = float(loss_dict["div_scale"].detach())
        div_contribution_value = float(loss_dict["div_contribution"].detach())
        risk_proxy_value = float(loss_dict["risk_proxy"].detach())
        diagnostics = _extract_step_diagnostics(model=model, loss_dict=loss_dict, step_idx=step_idx)
        div_eps_value = float(diagnostics["divergence_eps"])
        div_term_min = div_term_value if div_term_min is None else min(div_term_min, div_term_value)
        div_term_max = div_term_value if div_term_max is None else max(div_term_max, div_term_value)
        div_eps_min = div_eps_value if div_eps_min is None else min(div_eps_min, div_eps_value)
        div_eps_max = div_eps_value if div_eps_max is None else max(div_eps_max, div_eps_value)
        negative_div_steps += int(div_term_value < 0.0)
        if "dc_weight_min" in diagnostics:
            value = float(diagnostics["dc_weight_min"])
            dc_weight_min = value if dc_weight_min is None else min(dc_weight_min, value)
        if "dc_weight_max" in diagnostics:
            value = float(diagnostics["dc_weight_max"])
            dc_weight_max = value if dc_weight_max is None else max(dc_weight_max, value)
        if getattr(args, "save_step_debug", False) and (
            getattr(args, "step_debug_limit", None) is None
            or len(step_diagnostics) < int(args.step_debug_limit)
        ):
            step_diagnostics.append(diagnostics)
        stats.update(
            {
                "loss": loss_value,
                "data_term": data_term_value,
                "div_term": div_term_value,
                "div_scale": div_scale_value,
                "div_contribution": div_contribution_value,
                "risk_proxy": risk_proxy_value,
            }
        )
        progress.set_postfix(
            loss=f"{loss_value:.6f}",
            avg_loss=f"{stats.totals['loss'] / stats.count:.6f}",
            data=f"{data_term_value:.6f}",
            div=f"{div_term_value:.6f}",
            divc=f"{div_contribution_value:.6f}",
            eps=f"{div_eps_value:.2e}",
            refresh=False,
        )

    out = stats.averages()
    out["num_steps"] = stats.count
    out["observed_target_in_train"] = float(observed_target_in_train)
    out["train_div_term_min"] = float("nan") if div_term_min is None else div_term_min
    out["train_div_term_max"] = float("nan") if div_term_max is None else div_term_max
    out["train_negative_div_steps"] = negative_div_steps
    out["train_divergence_eps_min"] = float("nan") if div_eps_min is None else div_eps_min
    out["train_divergence_eps_max"] = float("nan") if div_eps_max is None else div_eps_max
    out["train_dc_weight_min"] = float("nan") if dc_weight_min is None else dc_weight_min
    out["train_dc_weight_max"] = float("nan") if dc_weight_max is None else dc_weight_max
    out["step_diagnostics"] = step_diagnostics
    return out


def validate(
    model: TemporalNormUnet,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, float]:
    model.eval()
    recon_stats = RunningStats()
    risk_stats = RunningStats()

    with torch.no_grad():
        progress = tqdm(loader, desc="val", leave=False)
        for batch_idx, batch in enumerate(progress):
            if args.max_val_batches is not None and batch_idx >= args.max_val_batches:
                break
            batch = move_to_device(batch, device)
            batch = maybe_center_crop_batch(batch, crop_height=args.crop_height, crop_width=args.crop_width)

            prediction = model(batch["zf"], maps=batch.get("maps"), mask=batch.get("mask"))
            metrics = summarize_reconstruction_metrics(
                prediction,
                batch["target_rss"],
                include_ssim=args.include_ssim,
            )
            recon_stats.update(metrics)
            postfix = {
                "nmse": f"{metrics['nmse']:.6f}",
                "avg_nmse": f"{recon_stats.totals['nmse'] / recon_stats.count:.6f}",
            }

            if args.compute_val_risk:
                loss_dict = _compute_train_loss(model, batch, args)
                risk_stats.update(
                    {
                        "val_risk_proxy": float(loss_dict["risk_proxy"]),
                        "val_data_term": float(loss_dict["data_term"]),
                        "val_div_term": float(loss_dict["div_term"]),
                        "val_div_scale": float(loss_dict["div_scale"]),
                        "val_div_contribution": float(loss_dict["div_contribution"]),
                    }
                )
                postfix["risk"] = f"{float(loss_dict['risk_proxy']):.6f}"
            progress.set_postfix(postfix, refresh=False)

    out = recon_stats.averages()
    out.update(risk_stats.averages())
    out["num_batches"] = recon_stats.count
    return out


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    configure_torch()
    set_random_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = ensure_dir(args.output_dir)

    train_dataset = _make_dataset(
        args,
        split=args.train_split,
        deterministic_masks=args.train_deterministic_masks,
        return_target=False,
    )
    val_dataset = _make_dataset(
        args,
        split=args.val_split,
        deterministic_masks=args.val_deterministic_masks,
        return_target=True,
    )
    train_loader = _make_loader(train_dataset, args, shuffle=True)
    val_loader = _make_loader(val_dataset, args, shuffle=False)

    model = TemporalNormUnet(
        num_frames=args.window_size,
        chans=args.chans,
        num_pools=args.num_pools,
        drop_prob=args.drop_prob,
        residual=not args.no_residual,
        output_mode="all_frames",
        num_unrolls=args.num_unrolls,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    config = vars(args).copy()
    config["device"] = str(device)
    config["train_dataset_len"] = len(train_dataset)
    config["val_dataset_len"] = len(val_dataset)
    config["num_parameters"] = count_trainable_parameters(model)
    save_json(output_dir / "config.json", config)

    history: list[Dict[str, Any]] = []
    debug_history: list[Dict[str, Any]] = []
    best_val_nmse = float("inf")

    for epoch_idx in range(args.epochs):
        train_stats = train_one_epoch(model, train_loader, optimizer, device, args)
        val_stats = validate(model, val_loader, device, args)

        epoch_record = {
            "epoch": epoch_idx,
            "train_loss": train_stats["loss"],
            "train_data_term": train_stats["data_term"],
            "train_div_term": train_stats["div_term"],
            "train_div_scale": train_stats["div_scale"],
            "train_div_contribution": train_stats["div_contribution"],
            "train_risk_proxy": train_stats["risk_proxy"],
            "val_nmse": val_stats["nmse"],
            "val_nrmse": val_stats["nrmse"],
            "val_psnr": val_stats["psnr"],
            "val_frame_diff_nmse": val_stats["frame_diff_nmse"],
            "train_steps": train_stats["num_steps"],
            "val_batches": val_stats["num_batches"],
            "observed_target_in_train": bool(train_stats["observed_target_in_train"]),
        }
        if "ssim" in val_stats:
            epoch_record["val_ssim"] = val_stats["ssim"]
        if "val_risk_proxy" in val_stats:
            epoch_record["val_risk_proxy"] = val_stats["val_risk_proxy"]
            epoch_record["val_data_term"] = val_stats["val_data_term"]
            epoch_record["val_div_term"] = val_stats["val_div_term"]
            epoch_record["val_div_scale"] = val_stats["val_div_scale"]
            epoch_record["val_div_contribution"] = val_stats["val_div_contribution"]
        for key in (
            "train_div_term_min",
            "train_div_term_max",
            "train_negative_div_steps",
            "train_divergence_eps_min",
            "train_divergence_eps_max",
            "train_dc_weight_min",
            "train_dc_weight_max",
        ):
            epoch_record[key] = train_stats[key]
        history.append(epoch_record)
        if getattr(args, "save_step_debug", False):
            debug_history.append(
                {
                    "epoch": epoch_idx,
                    "train_steps": train_stats["num_steps"],
                    "step_diagnostics": train_stats.get("step_diagnostics", []),
                }
            )

        print(
            f"[true-ensure] epoch={epoch_idx} "
            f"train_loss={epoch_record['train_loss']:.6f} "
            f"train_risk={epoch_record['train_risk_proxy']:.6f} "
            f"val_nmse={epoch_record['val_nmse']:.6f} "
            f"neg_div_steps={int(epoch_record['train_negative_div_steps'])}"
        )

        if val_stats["nmse"] < best_val_nmse:
            best_val_nmse = val_stats["nmse"]
            save_checkpoint(
                output_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch_idx,
                history=history,
                config=config,
                extra={"best_val_nmse": best_val_nmse},
            )

        if (epoch_idx + 1) % args.save_every == 0:
            save_checkpoint(
                output_dir / f"epoch_{epoch_idx:03d}.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch_idx,
                history=history,
                config=config,
                extra={"best_val_nmse": best_val_nmse},
            )

        save_json(output_dir / "history.json", {"history": history, "best_val_nmse": best_val_nmse})
        if getattr(args, "save_step_debug", False):
            save_json(output_dir / "debug_history.json", {"history": debug_history, "best_val_nmse": best_val_nmse})

    summary = {
        "best_val_nmse": best_val_nmse,
        "history": history,
        "output_dir": str(output_dir),
        "train_dataset_returns_target": bool(train_dataset.return_target),
        "val_dataset_returns_target": bool(val_dataset.return_target),
        "compute_val_risk": bool(args.compute_val_risk),
    }
    save_json(output_dir / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    run_experiment(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
