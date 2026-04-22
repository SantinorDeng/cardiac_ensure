#!/usr/bin/env python3
"""
Precompute Bernoulli-Gaussian sampling density statistics.

The cardiac ENSURE loss needs both the per-line sampling probability and a
stable estimate of the empirical sampling density used by density weighting.
This script writes one .npz file per (H, W, R, sigma_mask) configuration.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence, Tuple

import h5py
import numpy as np


def bernoulli_gaussian_line_prob(
    width: int,
    acceleration: float,
    sigma_mask: float = 0.18,
    max_prob: float = 0.95,
    tol: float = 1e-6,
    max_iter: int = 80,
) -> np.ndarray:
    """Return a Gaussian-shaped Bernoulli probability over ky lines."""
    if width <= 0:
        raise ValueError(f"width must be positive, got {width}")
    if acceleration <= 0:
        raise ValueError(f"acceleration must be positive, got {acceleration}")
    if sigma_mask <= 0:
        raise ValueError(f"sigma_mask must be positive, got {sigma_mask}")
    if not (0 < max_prob <= 1):
        raise ValueError(f"max_prob must be in (0, 1], got {max_prob}")

    target_rate = 1.0 / float(acceleration)
    if target_rate > max_prob + tol:
        raise ValueError(
            "Target sampling rate is larger than max_prob; lower acceleration "
            f"or increase max_prob. target={target_rate:.4f}, max_prob={max_prob:.4f}"
        )

    ky = np.linspace(-1.0, 1.0, int(width), dtype=np.float64)
    base = np.exp(-(ky**2) / (2.0 * float(sigma_mask) ** 2))

    lo = 0.0
    hi = max_prob / max(float(base.max()), np.finfo(np.float64).eps)
    while np.minimum(max_prob, hi * base).mean() < target_rate:
        hi *= 2.0

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        prob = np.minimum(max_prob, mid * base)
        if prob.mean() < target_rate:
            lo = mid
        else:
            hi = mid
        if abs(float(prob.mean()) - target_rate) < tol:
            break

    prob = np.minimum(max_prob, hi * base)
    return prob.astype(np.float32)


def sample_density(
    line_prob: np.ndarray,
    num_samples: int = 1024,
    seed: int = 0,
    density_floor: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Draw masks and return empirical density and inverse square-root weights."""
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    if density_floor <= 0:
        raise ValueError(f"density_floor must be positive, got {density_floor}")

    rng = np.random.default_rng(int(seed))
    draws = rng.random((int(num_samples), int(line_prob.shape[0])), dtype=np.float32)
    masks = draws < line_prob[None, :]
    empirical = masks.mean(axis=0).astype(np.float32)
    clipped = np.maximum(empirical, float(density_floor))
    inv_sqrt = (1.0 / np.sqrt(clipped)).astype(np.float32)
    return empirical, inv_sqrt


def density_stats_filename(
    height: int,
    width: int,
    acceleration: float,
    sigma_mask: float,
    num_samples: int,
) -> str:
    r_token = f"{float(acceleration):g}".replace(".", "p")
    s_token = f"{float(sigma_mask):g}".replace(".", "p")
    return f"density_H{height}_W{width}_R{r_token}_sigma{s_token}_N{num_samples}.npz"


def save_density_stats(
    output_dir: Path,
    shape: Tuple[int, int],
    acceleration: float,
    sigma_mask: float,
    num_samples: int = 1024,
    seed: int = 0,
    max_prob: float = 0.95,
    density_floor: float = 1e-3,
    overwrite: bool = False,
) -> Path:
    """Compute and save one density-statistics file."""
    height, width = int(shape[0]), int(shape[1])
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / density_stats_filename(
        height=height,
        width=width,
        acceleration=acceleration,
        sigma_mask=sigma_mask,
        num_samples=num_samples,
    )
    if out_path.exists() and not overwrite:
        return out_path

    line_prob = bernoulli_gaussian_line_prob(
        width=width,
        acceleration=acceleration,
        sigma_mask=sigma_mask,
        max_prob=max_prob,
    )
    empirical_density, inv_sqrt_density = sample_density(
        line_prob=line_prob,
        num_samples=num_samples,
        seed=seed,
        density_floor=density_floor,
    )

    np.savez_compressed(
        out_path,
        shape=np.asarray([height, width], dtype=np.int32),
        acceleration=np.asarray(float(acceleration), dtype=np.float32),
        sigma_mask=np.asarray(float(sigma_mask), dtype=np.float32),
        max_prob=np.asarray(float(max_prob), dtype=np.float32),
        target_rate=np.asarray(1.0 / float(acceleration), dtype=np.float32),
        empirical_rate=np.asarray(float(empirical_density.mean()), dtype=np.float32),
        num_samples=np.asarray(int(num_samples), dtype=np.int32),
        seed=np.asarray(int(seed), dtype=np.int64),
        density_floor=np.asarray(float(density_floor), dtype=np.float32),
        line_prob=line_prob,
        empirical_density=empirical_density,
        inv_sqrt_density=inv_sqrt_density,
    )
    return out_path


def discover_shapes(input_root: Path, splits: Sequence[str] | None = None) -> List[Tuple[int, int]]:
    """Scan h5 files and return unique image shapes."""
    roots: List[Path]
    if splits:
        roots = [input_root / split for split in splits]
    else:
        roots = [input_root]
        for split in ("train", "val", "test"):
            if (input_root / split).is_dir():
                roots.append(input_root / split)

    shapes = set()
    for root in roots:
        if not root.exists():
            continue
        for fname in sorted(root.glob("*.h5")):
            with h5py.File(str(fname), "r") as hf:
                if "kspace" not in hf:
                    continue
                shape = hf["kspace"].shape
                if len(shape) < 2:
                    continue
                shapes.add((int(shape[-2]), int(shape[-1])))
    return sorted(shapes)


def parse_shape_tokens(tokens: Sequence[str]) -> List[Tuple[int, int]]:
    """Parse repeated HxW or H W shape arguments."""
    if not tokens:
        return []
    if len(tokens) == 2 and all(token.isdigit() for token in tokens):
        return [(int(tokens[0]), int(tokens[1]))]

    shapes: List[Tuple[int, int]] = []
    for token in tokens:
        clean = token.lower().replace(",", "x")
        parts = [p for p in clean.split("x") if p]
        if len(parts) != 2:
            raise ValueError(f"Could not parse shape token: {token}")
        shapes.append((int(parts[0]), int(parts[1])))
    return shapes


'''
python precompute_density_stats.py \
    --output-root /home/dengyipin/CMR2025/cmr001/density_stats \
    --input-root /home/dengyipin/CMR2025/cmr001 \
    --splits train val \
    --accelerations 4.0 \
    --sigma-mask 0.18 \
    --num-samples 1024 \
    --seed 0 \
    --max-prob 0.95 \
    --density-floor 1e-3 \
    --overwrite
'''
def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, default=None)
    parser.add_argument("--splits", nargs="*", default=None)
    parser.add_argument("--shape", nargs="*", default=None, help="Either H W or repeated HxW tokens.")
    parser.add_argument("--accelerations", nargs="+", type=float, default=[4.0, 8.0])
    parser.add_argument("--sigma-mask", nargs="+", type=float, default=[0.18])
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-prob", type=float, default=0.95)
    parser.add_argument("--density-floor", type=float, default=1e-3)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)

    shapes = parse_shape_tokens(args.shape or [])
    if args.input_root is not None:
        shapes.extend(discover_shapes(args.input_root, splits=args.splits))
    shapes = sorted(set(shapes))
    if not shapes:
        raise SystemExit("No shapes provided or discovered. Use --shape or --input-root.")

    written: List[Path] = []
    for shape in shapes:
        for acceleration in args.accelerations:
            for sigma_mask in args.sigma_mask:
                out_path = save_density_stats(
                    output_dir=args.output_root,
                    shape=shape,
                    acceleration=acceleration,
                    sigma_mask=sigma_mask,
                    num_samples=args.num_samples,
                    seed=args.seed,
                    max_prob=args.max_prob,
                    density_floor=args.density_floor,
                    overwrite=args.overwrite,
                )
                written.append(out_path)
                print(f"wrote {out_path}")

    print(f"done: {len(written)} density files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
