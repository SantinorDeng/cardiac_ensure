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
from cardiac_ensure.models import TemporalNormUnet
from cardiac_ensure.scripts.eval_metrics import mean_nmse, summarize_reconstruction_metrics
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
    select_frame_mode,
    set_random_seed,
)
"""
 python /home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/scripts/train_supervised_baseline.py \
  --data-root /home/dengyipin/CMR2025/cmr001 \
  --preproc-root /home/dengyipin/CMR2025/cmr001/preproc_c \
  --density-root /home/dengyipin/CMR2025/cmr001/density_stats \
  --train-split train \
  --val-split val \
  --output-dir /home/dengyipin/project/gsure-diffusion-mri/cardiac_ensure/outputs/supervised_r4_w5_unroll3 \
  --acceleration 4.0 \
  --sigma-mask 0.18 \
  --window-size 5 \
  --stride 1 \
  --window-mode centered \
  --frame-mode all \
  --center-slice-fraction 1.0 \
  --epochs 30 \
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
  --val-deterministic-masks
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
    parser.add_argument("--frame-mode", choices=("all", "center"), default="all")
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
    return parser


def _make_dataset(args: argparse.Namespace, split: str, deterministic_masks: bool) -> CardiacCineENSUREDataset:
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
        return_target=True,
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


def _forward_supervised_loss(
    model: TemporalNormUnet,
    batch: Dict[str, Any],
    frame_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if "target_rss" not in batch:
        raise KeyError("Supervised training requires target_rss from the dataset.")
    prediction = model(batch["zf"], maps=batch.get("maps"), mask=batch.get("mask"))
    prediction = select_frame_mode(prediction, frame_mode)
    target = select_frame_mode(batch["target_rss"], frame_mode)
    loss = mean_nmse(prediction, target)
    return prediction, target, loss


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

    progress = tqdm(loader, desc="train", leave=False)
    for step_idx, batch in enumerate(progress):
        if args.max_train_steps is not None and step_idx >= args.max_train_steps:
            break

        batch = move_to_device(batch, device)
        batch = maybe_center_crop_batch(batch, crop_height=args.crop_height, crop_width=args.crop_width)
        observed_target_in_train = observed_target_in_train or ("target_rss" in batch)

        _, _, loss = _forward_supervised_loss(model, batch, args.frame_mode)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        loss_value = float(loss.detach())
        stats.update({"loss": loss_value})
        progress.set_postfix(
            loss=f"{loss_value:.6f}",
            avg_loss=f"{stats.totals['loss'] / stats.count:.6f}",
            refresh=False,
        )

    out = stats.averages()
    out["num_steps"] = stats.count
    out["observed_target_in_train"] = float(observed_target_in_train)
    return out


def validate(
    model: TemporalNormUnet,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, float]:
    model.eval()
    stats = RunningStats()

    with torch.no_grad():
        progress = tqdm(loader, desc="val", leave=False)
        for batch_idx, batch in enumerate(progress):
            if args.max_val_batches is not None and batch_idx >= args.max_val_batches:
                break
            batch = move_to_device(batch, device)
            batch = maybe_center_crop_batch(batch, crop_height=args.crop_height, crop_width=args.crop_width)
            prediction = model(batch["zf"], maps=batch.get("maps"), mask=batch.get("mask"))
            prediction = select_frame_mode(prediction, args.frame_mode)
            target = select_frame_mode(batch["target_rss"], args.frame_mode)
            metrics = summarize_reconstruction_metrics(
                prediction,
                target,
                include_ssim=args.include_ssim,
            )
            stats.update(metrics)
            progress.set_postfix(
                nmse=f"{metrics['nmse']:.6f}",
                avg_nmse=f"{stats.totals['nmse'] / stats.count:.6f}",
                refresh=False,
            )

    out = stats.averages()
    out["num_batches"] = stats.count
    return out


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    configure_torch()
    set_random_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = ensure_dir(args.output_dir)

    train_dataset = _make_dataset(args, args.train_split, args.train_deterministic_masks)
    val_dataset = _make_dataset(args, args.val_split, args.val_deterministic_masks)
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
    best_val_nmse = float("inf")

    for epoch_idx in range(args.epochs):
        train_stats = train_one_epoch(model, train_loader, optimizer, device, args)
        val_stats = validate(model, val_loader, device, args)

        epoch_record = {
            "epoch": epoch_idx,
            "train_loss": train_stats["loss"],
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
        history.append(epoch_record)

        print(
            f"[supervised] epoch={epoch_idx} "
            f"train_loss={epoch_record['train_loss']:.6f} "
            f"val_nmse={epoch_record['val_nmse']:.6f} "
            f"val_psnr={epoch_record['val_psnr']:.3f}"
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

    summary = {
        "best_val_nmse": best_val_nmse,
        "history": history,
        "output_dir": str(output_dir),
        "train_dataset_returns_target": bool(train_dataset.return_target),
        "val_dataset_returns_target": bool(val_dataset.return_target),
    }
    save_json(output_dir / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    run_experiment(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
