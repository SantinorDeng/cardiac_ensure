#!/usr/bin/env python3
"""
Build sidecar preprocessing files for cardiac cine ENSURE data.

This script intentionally does not rewrite the source h5 files. For each input
volume it writes a mirrored ``*.preproc.h5`` sidecar containing per-slice static
sensitivity maps, normalization constants, and native noise estimates.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import h5py
import numpy as np


EPS = 1e-8


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
    """Estimate simple static coil maps from time-averaged fully sampled data."""
    ref_kspace = np.asarray(kspace_tchw).mean(axis=0)
    coil_images = ifft2c(ref_kspace)
    rss = np.sqrt(np.sum(np.abs(coil_images) ** 2, axis=0, keepdims=True))
    maps = coil_images / np.maximum(rss, eps)
    return maps.astype(np.complex64)


def _frame_to_bart(kspace_chw: np.ndarray) -> np.ndarray:
    return np.asarray(kspace_chw).transpose(1, 2, 0)[:, :, None, :].astype(np.complex64)


def _extract_single_bart_map(sens_out: np.ndarray) -> np.ndarray:
    sens_out = np.asarray(sens_out)
    if sens_out.ndim == 5:
        sens_out = sens_out[:, :, :, :, 0]
    if sens_out.ndim != 4:
        raise ValueError(f"Unexpected BART ecalib output shape: {sens_out.shape}")
    return sens_out[:, :, 0, :].transpose(2, 0, 1).astype(np.complex64)


def load_bart(
    bart_toolbox_path: str | None = None,
    bart_python_path: str | None = None,
) -> Callable:
    """Load BART's Python wrapper without assuming a fixed install location."""
    candidates: List[str] = []
    if bart_python_path:
        candidates.append(bart_python_path)
    toolbox = bart_toolbox_path or os.environ.get("BART_TOOLBOX_PATH") or os.environ.get("TOOLBOX_PATH")
    if toolbox:
        candidates.append(str(Path(toolbox) / "python"))

    for candidate in candidates:
        if candidate and candidate not in sys.path:
            sys.path.append(candidate)

    from bart import bart

    return bart


def estimate_maps_bart(
    kspace_tchw: np.ndarray,
    bart_fn: Callable,
    ecalib_crop: float = 0.0,
    ecalib_cmd: str | None = None,
) -> np.ndarray:
    """Estimate ESPIRiT maps with BART ecalib from time-averaged k-space."""
    ref_kspace = np.asarray(kspace_tchw).mean(axis=0).astype(np.complex64)
    cmd = ecalib_cmd or f"ecalib -m1 -c{float(ecalib_crop)}"
    return _extract_single_bart_map(bart_fn(1, cmd, _frame_to_bart(ref_kspace)))


def compute_norm_scale(
    hf: h5py.File,
    slice_idx: int,
    kspace_tchw: np.ndarray,
    norm_source: str = "reconstruction_rss",
    percentile: float = 99.0,
) -> Tuple[float, str]:
    """Return a per-slice scale from stored RSS if available, else from k-space."""
    if norm_source == "reconstruction_rss" and "reconstruction_rss" in hf:
        rss = np.asarray(hf["reconstruction_rss"][:, slice_idx], dtype=np.float32)
        scale = float(np.percentile(np.abs(rss), percentile))
        return max(scale, EPS), "reconstruction_rss"

    coil_images = ifft2c(kspace_tchw)
    rss = np.sqrt(np.sum(np.abs(coil_images) ** 2, axis=1))
    scale = float(np.percentile(np.abs(rss), percentile))
    return max(scale, EPS), "rss_from_kspace"


def compute_noise_stats(
    kspace_tchw: np.ndarray,
    corner_fraction: float = 0.08,
) -> Tuple[float, float, float]:
    """Estimate native complex k-space noise from the four spatial corners."""
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
        return 0.0, 0.0, 0.0
    var_real = float(np.var(noise.real, ddof=1))
    var_imag = float(np.var(noise.imag, ddof=1))
    sigma2_complex = var_real + var_imag
    return sigma2_complex, var_real, var_imag


def center_slice_indices(num_slices: int, fraction: float = 1.0) -> np.ndarray:
    """Return the central slice subset used by pilot experiments."""
    if not (0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    keep = max(1, int(round(num_slices * fraction)))
    start = (num_slices - keep) // 2
    return np.arange(start, start + keep, dtype=np.int32)


def iter_h5_files(input_root: Path, splits: Sequence[str] | None = None) -> Iterable[Tuple[str, Path]]:
    """Yield (split, h5_path) pairs from a root or split directory."""
    if splits:
        for split in splits:
            split_root = input_root / split
            if not split_root.exists():
                continue
            for fname in sorted(split_root.glob("*.h5")):
                yield split, fname
        return

    found_split_dirs = False
    for split in ("train", "val", "test"):
        split_root = input_root / split
        if split_root.is_dir():
            found_split_dirs = True
            for fname in sorted(split_root.glob("*.h5")):
                yield split, fname
    if not found_split_dirs:
        for fname in sorted(input_root.glob("*.h5")):
            yield "", fname


def output_path_for(output_root: Path, split: str, source_path: Path) -> Path:
    out_dir = output_root / split if split else output_root
    return out_dir / f"{source_path.stem}.preproc.h5"


def create_dataset_kwargs(compression: str | None) -> Dict[str, object]:
    if compression is None:
        return {}
    compression = compression.strip().lower()
    if compression in ("", "none", "null", "false", "0"):
        return {}
    return {"compression": compression}


def process_file(
    source_path: Path,
    output_path: Path,
    map_method: str = "rss",
    norm_source: str = "reconstruction_rss",
    norm_percentile: float = 99.0,
    corner_fraction: float = 0.08,
    center_slice_fraction: float = 1.0,
    compression: str | None = "gzip",
    overwrite: bool = False,
    bart_fn: Callable | None = None,
    bart_ecalib_crop: float = 0.0,
    bart_ecalib_cmd: str | None = None,
) -> Path:
    """Process one source h5 file into one sidecar h5 file."""
    if output_path.exists() and not overwrite:
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(str(source_path), "r") as hf:
        if "kspace" not in hf:
            raise KeyError(f"Missing kspace dataset in {source_path}")
        kshape = hf["kspace"].shape
        if len(kshape) == 5:
            num_t, num_slices, num_coils, height, width = map(int, kshape)
        elif len(kshape) == 4:
            num_slices, num_coils, height, width = map(int, kshape)
            num_t = 1
        else:
            raise ValueError(f"Unsupported kspace shape in {source_path}: {kshape}")

        selected = center_slice_indices(num_slices, center_slice_fraction)
        processed_slice_mask = np.zeros((num_slices,), dtype=np.uint8)
        processed_slice_mask[selected] = 1

        dset_kwargs = create_dataset_kwargs(compression)
        with h5py.File(str(output_path), "w") as out:
            if map_method != "none":
                maps_ds = out.create_dataset(
                    "maps",
                    shape=(num_slices, num_coils, height, width),
                    dtype=np.complex64,
                    chunks=(1, num_coils, height, width),
                    **dset_kwargs,
                )
            else:
                maps_ds = None

            norm_scale = out.create_dataset("norm_scale", shape=(num_slices,), dtype=np.float32)
            noise_sigma2 = out.create_dataset("noise_sigma2", shape=(num_slices,), dtype=np.float32)
            noise_sigma2_raw = out.create_dataset("noise_sigma2_raw", shape=(num_slices,), dtype=np.float32)
            noise_var_real = out.create_dataset("noise_var_real", shape=(num_slices,), dtype=np.float32)
            noise_var_imag = out.create_dataset("noise_var_imag", shape=(num_slices,), dtype=np.float32)
            norm_source_used = out.create_dataset(
                "norm_source_used",
                shape=(num_slices,),
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
            out.create_dataset("processed_slice_mask", data=processed_slice_mask)

            norm_scale[...] = np.nan
            noise_sigma2[...] = np.nan
            noise_sigma2_raw[...] = np.nan
            noise_var_real[...] = np.nan
            noise_var_imag[...] = np.nan
            norm_source_used[...] = ""

            for key, value in hf.attrs.items():
                try:
                    out.attrs[f"source_attr_{key}"] = value
                except TypeError:
                    out.attrs[f"source_attr_{key}"] = str(value)
            out.attrs["source_file"] = str(source_path)
            out.attrs["source_shape"] = np.asarray(kshape, dtype=np.int32)
            out.attrs["map_method"] = map_method
            out.attrs["norm_source_requested"] = norm_source
            out.attrs["norm_percentile"] = float(norm_percentile)
            out.attrs["noise_sigma2_definition"] = "var(real) + var(imag), normalized by norm_scale**2"
            out.attrs["center_slice_fraction"] = float(center_slice_fraction)

            for slice_idx in selected.tolist():
                if len(kshape) == 5:
                    kspace_tchw = ensure_complex(hf["kspace"][:, slice_idx]).astype(np.complex64, copy=False)
                else:
                    kspace_tchw = ensure_complex(hf["kspace"][slice_idx])[None].astype(np.complex64, copy=False)

                scale, used_source = compute_norm_scale(
                    hf=hf,
                    slice_idx=slice_idx,
                    kspace_tchw=kspace_tchw,
                    norm_source=norm_source,
                    percentile=norm_percentile,
                )
                sigma2_raw, var_r_raw, var_i_raw = compute_noise_stats(
                    kspace_tchw=kspace_tchw,
                    corner_fraction=corner_fraction,
                )
                scale2 = max(scale * scale, EPS)

                norm_scale[slice_idx] = np.float32(scale)
                noise_sigma2_raw[slice_idx] = np.float32(sigma2_raw)
                noise_var_real[slice_idx] = np.float32(var_r_raw / scale2)
                noise_var_imag[slice_idx] = np.float32(var_i_raw / scale2)
                noise_sigma2[slice_idx] = np.float32(sigma2_raw / scale2)
                norm_source_used[slice_idx] = used_source

                if maps_ds is not None:
                    if map_method == "rss":
                        maps = estimate_maps_rss(kspace_tchw)
                    elif map_method == "bart":
                        if bart_fn is None:
                            raise RuntimeError("map_method='bart' requested but BART was not loaded")
                        maps = estimate_maps_bart(
                            kspace_tchw=kspace_tchw,
                            bart_fn=bart_fn,
                            ecalib_crop=bart_ecalib_crop,
                            ecalib_cmd=bart_ecalib_cmd,
                        )
                    else:
                        raise ValueError(f"Unsupported map_method: {map_method}")
                    maps_ds[slice_idx] = maps

    return output_path

"""
python preprocess_raw_cine.py \
  --input-root /home/dengyipin/CMR2025/cmr001 \
  --output-root /home/dengyipin/CMR2025/cmr001/preproc_c \
  --splits train val \
  --map-method bart \
  --norm-source reconstruction_rss \
  --norm-percentile 95.0 \
  --corner-fraction 0.08 \
  --center-slice-fraction 1.0 \
  --compression None \
  --overwrite \
"""
def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--splits", nargs="*", default=None)
    parser.add_argument("--map-method", choices=("rss", "bart", "none"), default="rss")
    parser.add_argument("--norm-source", choices=("reconstruction_rss", "rss_from_kspace"), default="reconstruction_rss")
    parser.add_argument("--norm-percentile", type=float, default=99.0)
    parser.add_argument("--corner-fraction", type=float, default=0.08)
    parser.add_argument("--center-slice-fraction", type=float, default=1.0)
    parser.add_argument("--compression", default="gzip")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--bart-toolbox-path", default=None)
    parser.add_argument("--bart-python-path", default=None)
    parser.add_argument("--bart-ecalib-crop", type=float, default=0.0)
    parser.add_argument("--bart-ecalib-cmd", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)

    bart_fn = None
    if args.map_method == "bart":
        bart_fn = load_bart(
            bart_toolbox_path=args.bart_toolbox_path,
            bart_python_path=args.bart_python_path,
        )

    files = list(iter_h5_files(args.input_root, splits=args.splits))
    if args.max_files is not None:
        files = files[: int(args.max_files)]
    if not files:
        raise SystemExit(f"No h5 files found under {args.input_root}")

    for idx, (split, source_path) in enumerate(files, start=1):
        out_path = output_path_for(args.output_root, split, source_path)
        process_file(
            source_path=source_path,
            output_path=out_path,
            map_method=args.map_method,
            norm_source=args.norm_source,
            norm_percentile=args.norm_percentile,
            corner_fraction=args.corner_fraction,
            center_slice_fraction=args.center_slice_fraction,
            compression=args.compression,
            overwrite=args.overwrite,
            bart_fn=bart_fn,
            bart_ecalib_crop=args.bart_ecalib_crop,
            bart_ecalib_cmd=args.bart_ecalib_cmd,
        )
        print(f"[{idx}/{len(files)}] wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
