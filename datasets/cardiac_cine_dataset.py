#!/usr/bin/env python3
"""
PyTorch dataset for cardiac dynamic ENSURE experiments.

The dataset keeps expensive, acquisition-level products in preprocessing
sidecars and performs sample-level operations online:

* time-window selection,
* Bernoulli-Gaussian per-frame mask realization,
* undersampling,
* zero-filled/adjoint image construction.

It deliberately does not return rho_ls; that belongs in the training/loss step.
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from .precompute_density_stats import bernoulli_gaussian_line_prob
except ImportError:  # pragma: no cover - useful when this file is copied alone.
    try:
        from precompute_density_stats import bernoulli_gaussian_line_prob
    except ImportError:
        def bernoulli_gaussian_line_prob(
            width: int,
            acceleration: float,
            sigma_mask: float = 0.18,
            max_prob: float = 0.95,
            tol: float = 1e-6,
            max_iter: int = 80,
        ) -> np.ndarray:
            target_rate = 1.0 / float(acceleration)
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
            return np.minimum(max_prob, hi * base).astype(np.float32)


EPS = 1e-8


@dataclass(frozen=True)
class CineWindowSample:
    fname: Path
    preproc_path: Optional[Path]
    volume_id: str
    slice_id: int
    center_frame: int
    frame_indices: Tuple[int, ...]
    frame_count: int
    num_slices: int
    height: int
    width: int


def ensure_complex(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr)
    if np.iscomplexobj(x):
        return x
    if x.ndim >= 1 and x.shape[-1] == 2:
        return x[..., 0] + 1j * x[..., 1]
    raise ValueError(f"Expected complex data or trailing real/imag dim, got shape={x.shape}, dtype={x.dtype}")


def ifft2c(kspace: np.ndarray) -> np.ndarray:
    shifted = np.fft.ifftshift(kspace, axes=(-2, -1))
    image = np.fft.ifft2(shifted, axes=(-2, -1), norm="ortho")
    return np.fft.fftshift(image, axes=(-2, -1))


def estimate_maps_rss(kspace_tchw: np.ndarray, eps: float = EPS) -> np.ndarray:
    ref_kspace = np.asarray(kspace_tchw).mean(axis=0)
    coil_images = ifft2c(ref_kspace)
    rss = np.sqrt(np.sum(np.abs(coil_images) ** 2, axis=0, keepdims=True))
    return (coil_images / np.maximum(rss, eps)).astype(np.complex64)


def estimate_noise_sigma2(kspace_tchw: np.ndarray, scale: float = 1.0, corner_fraction: float = 0.08) -> float:
    if not (0 < corner_fraction < 0.5):
        raise ValueError(f"corner_fraction must be in (0, 0.5), got {corner_fraction}")
    _, _, height, width = kspace_tchw.shape
    corner_h = max(1, int(round(height * corner_fraction)))
    corner_w = max(1, int(round(width * corner_fraction)))
    patches = [
        kspace_tchw[..., :corner_h, :corner_w],
        kspace_tchw[..., :corner_h, -corner_w:],
        kspace_tchw[..., -corner_h:, :corner_w],
        kspace_tchw[..., -corner_h:, -corner_w:],
    ]
    noise = np.concatenate([patch.reshape(-1) for patch in patches], axis=0)
    if noise.size <= 1:
        return 0.0
    sigma2_raw = float(np.var(noise.real, ddof=1) + np.var(noise.imag, ddof=1))
    return sigma2_raw / max(float(scale) * float(scale), EPS)


def adjoint_image(kspace_us: np.ndarray, maps_chw: np.ndarray) -> np.ndarray:
    coil_images = ifft2c(kspace_us)
    img = np.sum(coil_images * np.conj(maps_chw[None, ...]), axis=1)
    return img[:, None, ...].astype(np.complex64)


def center_slice_indices(num_slices: int, fraction: float = 1.0) -> List[int]:
    if not (0 < fraction <= 1.0):
        raise ValueError(f"center_slice_fraction must be in (0, 1], got {fraction}")
    keep = max(1, int(round(num_slices * fraction)))
    start = (num_slices - keep) // 2
    return list(range(start, start + keep))


def centered_frame_indices(center: int, frame_count: int, window_size: int) -> Tuple[int, ...]:
    if frame_count <= 0:
        raise ValueError(f"frame_count must be positive, got {frame_count}")
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    start = center - (window_size // 2)
    return tuple((start + offset) % frame_count for offset in range(window_size))


def sliding_frame_indices(start: int, frame_count: int, window_size: int) -> Tuple[int, ...]:
    if frame_count >= window_size:
        return tuple(range(start, start + window_size))
    return tuple((start + offset) % frame_count for offset in range(window_size))


def resolve_data_root(root: Path, split: Optional[str]) -> Tuple[Path, str]:
    if split and (root / split).is_dir():
        return root / split, split
    if split is None and root.name in {"train", "val", "test"}:
        return root, root.name
    return root, split or ""


def resolve_preproc_path(preproc_root: Optional[Path], split: str, fname: Path) -> Optional[Path]:
    if preproc_root is None:
        return None
    candidates = []
    if split:
        candidates.append(preproc_root / split / f"{fname.stem}.preproc.h5")
    candidates.append(preproc_root / f"{fname.stem}.preproc.h5")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else None


def stable_seed(*parts: object, modulo: int = 2**32) -> int:
    msg = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(msg).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % modulo


def make_mask_from_line_prob(
    line_prob: np.ndarray,
    height: int,
    num_frames: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    draws = rng.random((num_frames, int(line_prob.shape[0])), dtype=np.float32)
    line_mask = (draws < line_prob[None, :]).astype(np.float32)
    mask = np.broadcast_to(line_mask[:, None, None, :], (num_frames, 1, height, line_prob.shape[0]))
    prob = np.broadcast_to(line_prob[None, None, None, :], mask.shape)
    return mask.astype(np.float32), prob.astype(np.float32)


def density_stats_filename_glob(height: int, width: int, acceleration: float, sigma_mask: float) -> str:
    r_token = f"{float(acceleration):g}".replace(".", "p")
    s_token = f"{float(sigma_mask):g}".replace(".", "p")
    return f"density_H{height}_W{width}_R{r_token}_sigma{s_token}_N*.npz"


def load_density_stats(
    density_root: Optional[Path],
    height: int,
    width: int,
    acceleration: float,
    sigma_mask: float,
) -> Dict[str, np.ndarray]:
    if density_root is None:
        return {}
    pattern = density_stats_filename_glob(height, width, acceleration, sigma_mask)
    matches = sorted(density_root.glob(pattern))
    if not matches:
        return {}
    path = matches[-1]
    with np.load(path) as npz:
        return {
            "density_line_prob": np.asarray(npz["line_prob"], dtype=np.float32),
            "empirical_density": np.asarray(npz["empirical_density"], dtype=np.float32),
            "inv_sqrt_density": np.asarray(npz["inv_sqrt_density"], dtype=np.float32),
            "density_stats_path": np.asarray(str(path)),
        }


class CardiacCineENSUREDataset(Dataset):
    """Return dynamic cardiac ENSURE training samples."""

    def __init__(
        self,
        root: str | Path,
        split: Optional[str] = None,
        preproc_root: str | Path | None = None,
        density_root: str | Path | None = None,
        acceleration: float = 4.0,
        sigma_mask: float = 0.18,
        window_size: int = 8,
        stride: int = 1,
        window_mode: str = "centered",
        center_slice_fraction: float = 1.0,
        deterministic_masks: bool = True,
        mask_seed: int = 0,
        max_prob: float = 0.95,
        normalize: bool = True,
        return_target: bool = True,
        require_preproc: bool = False,
        cache_maps: bool = True,
    ) -> None:
        self.root = Path(root)
        self.data_root, self.split = resolve_data_root(self.root, split)
        self.preproc_root = Path(preproc_root) if preproc_root is not None else None
        self.density_root = Path(density_root) if density_root is not None else None
        self.acceleration = float(acceleration)
        self.sigma_mask = float(sigma_mask)
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.window_mode = window_mode
        self.center_slice_fraction = float(center_slice_fraction)
        self.deterministic_masks = bool(deterministic_masks)
        self.mask_seed = int(mask_seed)
        self.max_prob = float(max_prob)
        self.normalize = bool(normalize)
        self.return_target = bool(return_target)
        self.require_preproc = bool(require_preproc)
        self.cache_maps = bool(cache_maps)
        self._map_cache: Dict[Tuple[Path, int], np.ndarray] = {}
        self._density_cache: Dict[Tuple[int, int, float, float], Dict[str, np.ndarray]] = {}

        if self.window_mode not in {"centered", "sliding"}:
            raise ValueError("window_mode must be 'centered' or 'sliding'")
        if self.stride <= 0:
            raise ValueError(f"stride must be positive, got {self.stride}")

        self.samples = self._build_samples()

    def _build_samples(self) -> List[CineWindowSample]:
        files = sorted(self.data_root.glob("*.h5"))
        if not files:
            raise FileNotFoundError(f"No h5 files found under {self.data_root}")

        samples: List[CineWindowSample] = []
        for fname in files:
            preproc_path = resolve_preproc_path(self.preproc_root, self.split, fname)
            if self.require_preproc and (preproc_path is None or not preproc_path.exists()):
                raise FileNotFoundError(f"Missing preprocessing sidecar for {fname}: {preproc_path}")

            with h5py.File(str(fname), "r") as hf:
                if "kspace" not in hf:
                    continue
                kshape = hf["kspace"].shape
                if len(kshape) == 5:
                    frame_count, num_slices, _, height, width = map(int, kshape)
                elif len(kshape) == 4:
                    num_slices, _, height, width = map(int, kshape)
                    frame_count = 1
                else:
                    raise ValueError(f"Unsupported kspace shape in {fname}: {kshape}")

            selected_slices = center_slice_indices(num_slices, self.center_slice_fraction)
            volume_id = fname.stem
            for slice_id in selected_slices:
                if self.window_mode == "centered":
                    centers = range(0, frame_count, self.stride)
                    for center in centers:
                        frames = centered_frame_indices(center, frame_count, self.window_size)
                        samples.append(
                            CineWindowSample(
                                fname=fname,
                                preproc_path=preproc_path,
                                volume_id=volume_id,
                                slice_id=slice_id,
                                center_frame=center,
                                frame_indices=frames,
                                frame_count=frame_count,
                                num_slices=num_slices,
                                height=height,
                                width=width,
                            )
                        )
                else:
                    if frame_count >= self.window_size:
                        starts = range(0, frame_count - self.window_size + 1, self.stride)
                    else:
                        starts = range(0, 1)
                    for start in starts:
                        frames = sliding_frame_indices(start, frame_count, self.window_size)
                        samples.append(
                            CineWindowSample(
                                fname=fname,
                                preproc_path=preproc_path,
                                volume_id=volume_id,
                                slice_id=slice_id,
                                center_frame=start,
                                frame_indices=frames,
                                frame_count=frame_count,
                                num_slices=num_slices,
                                height=height,
                                width=width,
                            )
                        )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _read_kspace_window(self, hf: h5py.File, sample: CineWindowSample) -> np.ndarray:
        kshape = hf["kspace"].shape
        if len(kshape) == 5:
            frames = [ensure_complex(hf["kspace"][frame, sample.slice_id]) for frame in sample.frame_indices]
            kspace = np.stack(frames, axis=0)
        else:
            frame = ensure_complex(hf["kspace"][sample.slice_id])
            kspace = np.stack([frame for _ in sample.frame_indices], axis=0)
        return kspace.astype(np.complex64, copy=False)

    def _read_target_window(self, hf: h5py.File, sample: CineWindowSample, scale: float) -> Optional[np.ndarray]:
        if not self.return_target or "reconstruction_rss" not in hf:
            return None
        if len(hf["reconstruction_rss"].shape) == 4:
            frames = [np.asarray(hf["reconstruction_rss"][frame, sample.slice_id], dtype=np.float32) for frame in sample.frame_indices]
            target = np.stack(frames, axis=0)
        elif len(hf["reconstruction_rss"].shape) == 3:
            frame = np.asarray(hf["reconstruction_rss"][sample.slice_id], dtype=np.float32)
            target = np.stack([frame for _ in sample.frame_indices], axis=0)
        else:
            return None
        if self.normalize:
            target = target / max(float(scale), EPS)
        return target[:, None, ...].astype(np.float32)

    def _fallback_scale(self, hf: h5py.File, sample: CineWindowSample) -> float:
        if "reconstruction_rss" in hf:
            if len(hf["reconstruction_rss"].shape) == 4:
                rss = np.asarray(hf["reconstruction_rss"][:, sample.slice_id], dtype=np.float32)
            elif len(hf["reconstruction_rss"].shape) == 3:
                rss = np.asarray(hf["reconstruction_rss"][sample.slice_id], dtype=np.float32)
            else:
                rss = None
            if rss is not None:
                scale = float(np.percentile(np.abs(rss), 99.0))
                if np.isfinite(scale) and scale > 0:
                    return scale

        attrs_max = float(hf.attrs.get("max", 0.0)) if "max" in hf.attrs else 0.0
        return attrs_max if attrs_max > 0 else 1.0

    def _read_preproc_values(self, sample: CineWindowSample) -> Tuple[Optional[np.ndarray], float, float, Dict[str, object], bool]:
        maps = None
        scale = 1.0
        noise_sigma2 = 0.0
        meta: Dict[str, object] = {}
        path = sample.preproc_path
        if path is None or not path.exists():
            return maps, scale, noise_sigma2, meta, False

        with h5py.File(str(path), "r") as hf:
            processed = True
            if "processed_slice_mask" in hf:
                processed = bool(np.asarray(hf["processed_slice_mask"][sample.slice_id]).item())
            if "norm_scale" in hf and processed:
                value = float(hf["norm_scale"][sample.slice_id])
                if np.isfinite(value) and value > 0:
                    scale = value
            if "noise_sigma2" in hf and processed:
                value = float(hf["noise_sigma2"][sample.slice_id])
                if np.isfinite(value) and value >= 0:
                    noise_sigma2 = value
            if "maps" in hf and processed:
                maps = np.asarray(hf["maps"][sample.slice_id], dtype=np.complex64)
            meta["preproc_path"] = str(path)
            meta["map_method"] = str(hf.attrs.get("map_method", "unknown"))
            if not processed:
                meta["preproc_slice_processed"] = False
        return maps, scale, noise_sigma2, meta, processed

    def _get_maps(self, sample: CineWindowSample, full_slice_kspace_tchw: np.ndarray, preproc_maps: Optional[np.ndarray]) -> np.ndarray:
        if preproc_maps is not None:
            return preproc_maps.astype(np.complex64, copy=False)
        cache_key = (sample.fname, sample.slice_id)
        if self.cache_maps and cache_key in self._map_cache:
            return self._map_cache[cache_key]
        maps = estimate_maps_rss(full_slice_kspace_tchw)
        if self.cache_maps:
            self._map_cache[cache_key] = maps
        return maps

    def _get_density_stats(self, height: int, width: int) -> Dict[str, np.ndarray]:
        key = (height, width, self.acceleration, self.sigma_mask)
        if key not in self._density_cache:
            self._density_cache[key] = load_density_stats(
                density_root=self.density_root,
                height=height,
                width=width,
                acceleration=self.acceleration,
                sigma_mask=self.sigma_mask,
            )
        return self._density_cache[key]

    def _rng_for_sample(self, sample: CineWindowSample, idx: int) -> np.random.Generator:
        if self.deterministic_masks:
            seed = stable_seed(
                self.mask_seed,
                sample.volume_id,
                sample.slice_id,
                sample.center_frame,
                sample.frame_indices,
                self.acceleration,
                self.sigma_mask,
            )
            return np.random.default_rng(seed)
        return np.random.default_rng()

    def __getitem__(self, idx: int) -> Mapping[str, object]:
        sample = self.samples[int(idx)]

        preproc_maps, scale, noise_sigma2, meta_from_preproc, preproc_processed = self._read_preproc_values(sample)
        with h5py.File(str(sample.fname), "r") as hf:
            kspace_fs = self._read_kspace_window(hf, sample)
            if self.normalize and not preproc_processed:
                scale = self._fallback_scale(hf, sample)
            if preproc_maps is None:
                if len(hf["kspace"].shape) == 5:
                    full_slice_kspace = ensure_complex(hf["kspace"][:, sample.slice_id]).astype(np.complex64, copy=False)
                else:
                    full_slice_kspace = ensure_complex(hf["kspace"][sample.slice_id])[None].astype(np.complex64, copy=False)
            else:
                full_slice_kspace = kspace_fs
            target = self._read_target_window(hf, sample, scale)

        maps = self._get_maps(sample, full_slice_kspace, preproc_maps)
        if not preproc_processed:
            noise_sigma2 = estimate_noise_sigma2(full_slice_kspace, scale=scale)

        if self.normalize:
            kspace_fs = kspace_fs / max(float(scale), EPS)

        density_stats = self._get_density_stats(sample.height, sample.width)
        if "density_line_prob" in density_stats:
            line_prob = density_stats["density_line_prob"]
        else:
            line_prob = bernoulli_gaussian_line_prob(
                width=sample.width,
                acceleration=self.acceleration,
                sigma_mask=self.sigma_mask,
                max_prob=self.max_prob,
            )

        rng = self._rng_for_sample(sample, int(idx))
        mask, mask_prob = make_mask_from_line_prob(
            line_prob=line_prob,
            height=sample.height,
            num_frames=self.window_size,
            rng=rng,
        )
        kspace_us = kspace_fs * mask
        zf = adjoint_image(kspace_us, maps)

        meta: Dict[str, object] = {
            "volume_id": sample.volume_id,
            "slice_id": int(sample.slice_id),
            "center_frame": int(sample.center_frame),
            "frame_indices": tuple(int(x) for x in sample.frame_indices),
            "frame_count": int(sample.frame_count),
            "source_file": str(sample.fname),
            "split": self.split,
            "norm_scale": float(scale),
            "acceleration": float(self.acceleration),
            "sigma_mask": float(self.sigma_mask),
            "window_mode": self.window_mode,
        }
        meta.update(meta_from_preproc)
        if "density_stats_path" in density_stats:
            meta["density_stats_path"] = str(density_stats["density_stats_path"])

        out: Dict[str, object] = {
            "kspace_fs": torch.from_numpy(kspace_fs.astype(np.complex64, copy=False)),
            "kspace_us": torch.from_numpy(kspace_us.astype(np.complex64, copy=False)),
            "mask": torch.from_numpy(mask),
            "mask_prob": torch.from_numpy(mask_prob),
            "maps": torch.from_numpy(maps.astype(np.complex64, copy=False)),
            "zf": torch.from_numpy(zf),
            "noise_sigma2": torch.tensor(float(noise_sigma2), dtype=torch.float32),
            "meta": meta,
        }
        if target is not None:
            out["target_rss"] = torch.from_numpy(target)
        if "empirical_density" in density_stats:
            out["empirical_density"] = torch.from_numpy(density_stats["empirical_density"])
        if "inv_sqrt_density" in density_stats:
            out["inv_sqrt_density"] = torch.from_numpy(density_stats["inv_sqrt_density"])
        return out


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-test CardiacCineENSUREDataset.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--preproc-root", type=Path, default=None)
    parser.add_argument("--density-root", type=Path, default=None)
    parser.add_argument("--acceleration", type=float, default=4.0)
    parser.add_argument("--sigma-mask", type=float, default=0.18)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--window-mode", choices=("centered", "sliding"), default="centered")
    parser.add_argument("--center-slice-fraction", type=float, default=1.0)
    parser.add_argument("--index", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    dataset = CardiacCineENSUREDataset(
        root=args.root,
        split=args.split,
        preproc_root=args.preproc_root,
        density_root=args.density_root,
        acceleration=args.acceleration,
        sigma_mask=args.sigma_mask,
        window_size=args.window_size,
        window_mode=args.window_mode,
        center_slice_fraction=args.center_slice_fraction,
    )
    sample = dataset[args.index]
    print(f"dataset length: {len(dataset)}")
    for key, value in sample.items():
        if torch.is_tensor(value):
            print(f"{key}: shape={tuple(value.shape)}, dtype={value.dtype}")
        else:
            print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
