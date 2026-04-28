"""NeuroCombat baseline for the ABIDE fixed-slice benchmark.

This script implements a statistical harmonization baseline for raw ABIDE-I T1
MRI slices. It deliberately reuses ``AbideSlicesDataset`` with the same fixed
subject-slice policy used by the raw site-probe benchmark, then treats each
pixel location in the deterministic 2D slice as a feature for NeuroCombat.

The saved outputs are intended to be passed to the same frozen site-probe and
preservation/distribution metric pipeline used for all benchmark methods.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


METHOD = "neurocombat"


def parse_out_hw(values: Iterable[int]) -> tuple[int, int]:
    values = tuple(int(v) for v in values)
    if len(values) == 1:
        return values[0], values[0]
    if len(values) == 2:
        return values[0], values[1]
    raise argparse.ArgumentTypeError("--out-hw expects one integer or two integers")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply NeuroCombat to deterministic ABIDE fixed slices."
    )
    parser.add_argument("--manifest-path", default="data/abide_manifest.csv")
    parser.add_argument("--splits-path", default="data/splits.json")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--out-dir", default="outputs/harmonized/neurocombat")
    parser.add_argument("--out-hw", nargs="+", type=int, default=(256, 256))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--valid-nonzero-frac",
        type=float,
        default=float(os.getenv("VALID_FG_FRAC", "0.02")),
        help="Matches the site-probe evaluation default slice foreground filter.",
    )
    parser.add_argument(
        "--fg-bbox-thr",
        type=float,
        default=float(os.getenv("FG_BBOX_THR", "0.02")),
        help="Matches the site-probe evaluation default foreground threshold.",
    )
    parser.add_argument(
        "--volume-cache-size",
        type=int,
        default=12,
        help="Number of normalized NIfTI volumes to keep in the dataset LRU cache.",
    )
    parser.add_argument(
        "--no-qc",
        action="store_true",
        help="Skip writing qc_raw_vs_neurocombat.png.",
    )
    return parser


def set_seed(seed: int) -> None:
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "Missing dependency 'torch'. Install the project runtime dependencies "
            "before running the NeuroCombat baseline."
        ) from exc

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def describe_array(name: str, array: np.ndarray) -> None:
    print(
        f"{name}: shape={array.shape} "
        f"min={float(array.min()):.6f} max={float(array.max()):.6f} "
        f"mean={float(array.mean()):.6f} std={float(array.std()):.6f}"
    )


def import_neurocombat():
    try:
        from neuroCombat import neuroCombat
    except ImportError as exc:
        raise ImportError(
            "Missing dependency 'neuroCombat'. Install it before running this baseline, "
            "for example with `pip install neuroCombat`."
        ) from exc
    return neuroCombat


def build_dataset(args: argparse.Namespace, out_hw: tuple[int, int]) -> Any:
    try:
        from harmonit.data.abide_slices_dataset import AbideSlicesDataset
    except ImportError as exc:
        raise ImportError(
            "Could not import harmonit.data.abide_slices_dataset. Run from the "
            "repository root with project dependencies installed."
        ) from exc

    dataset = AbideSlicesDataset(
        manifest_path=args.manifest_path,
        splits_path=args.splits_path,
        split=args.split,
        out_hw=out_hw,
        slice_mode="fixed",
        valid_nonzero_frac=args.valid_nonzero_frac,
        fg_bbox_thr=args.fg_bbox_thr,
        seed=args.seed,
        volume_cache_size=args.volume_cache_size,
        mask_mode="none",
        bg_suppress=True,
        input_mode="image",
    )

    # The benchmark artifact must be deterministic even when users have probe
    # training augmentation variables in their shell and request --split train.
    dataset.aug_affine = False
    return dataset


def extract_fixed_slices(
    dataset: Any,
    batch_size: int,
    num_workers: int,
    out_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    if len(dataset) == 0:
        raise ValueError("ABIDE split is empty; no fixed slices to harmonize")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=(num_workers > 0),
    )

    image_batches: list[np.ndarray] = []
    subject_ids: list[str] = []
    site_ids: list[int] = []
    slice_indices: list[int] = []

    for images, sites, subjects, slices in tqdm(loader, desc="Loading fixed ABIDE slices"):
        expected_shape = (images.shape[0], 1, out_hw[0], out_hw[1])
        if tuple(images.shape) != expected_shape:
            raise AssertionError(
                f"Expected batch image shape {expected_shape}, got {tuple(images.shape)}"
            )

        image_batches.append(images.cpu().numpy().astype(np.float32, copy=False))
        subject_ids.extend(str(subject_id) for subject_id in subjects)
        site_ids.extend(int(site_id) for site_id in sites.cpu().numpy().tolist())
        slice_indices.extend(int(slice_idx) for slice_idx in slices.cpu().numpy().tolist())

    raw_images = np.concatenate(image_batches, axis=0)
    subject_ids_arr = np.asarray(subject_ids, dtype=str)
    site_ids_arr = np.asarray(site_ids, dtype=np.int64)
    slice_indices_arr = np.asarray(slice_indices, dtype=np.int64)

    if raw_images.shape[0] != len(dataset):
        raise AssertionError(
            f"Loaded {raw_images.shape[0]} images, but dataset has {len(dataset)} samples"
        )
    if raw_images.shape[1:] != (1, out_hw[0], out_hw[1]):
        raise AssertionError(
            f"Expected per-sample image shape {(1, out_hw[0], out_hw[1])}, "
            f"got {raw_images.shape[1:]}"
        )
    if not np.isfinite(raw_images).all():
        raise AssertionError("Raw fixed slices contain NaN or inf values")

    return raw_images, subject_ids_arr, site_ids_arr, slice_indices_arr


def apply_neurocombat_to_pixels(raw_images: np.ndarray, site_ids: np.ndarray) -> np.ndarray:
    neuroCombat = import_neurocombat()

    n_images, channels, height, width = raw_images.shape
    if channels != 1:
        raise ValueError(f"Expected single-channel slices, got C={channels}")
    if n_images != len(site_ids):
        raise ValueError(f"images N={n_images} does not match site_ids N={len(site_ids)}")

    features = raw_images.reshape(n_images, height * width).astype(np.float64, copy=False)

    unique_sites = np.unique(site_ids)
    if unique_sites.size < 2:
        raise ValueError("NeuroCombat requires at least two site batches")

    # neuroCombat expects dat with shape [features, samples]. Our benchmark
    # tensors are [samples, 1, H, W], so flatten to [samples, H*W], transpose
    # only for the library call, and transpose back after harmonization.
    feature_std = features.std(axis=0)
    variable_feature_mask = feature_std > 1e-12
    n_constant = int((~variable_feature_mask).sum())
    if n_constant:
        print(f"Skipping {n_constant} constant pixel features during NeuroCombat.")
    if not variable_feature_mask.any():
        raise ValueError("All pixel features are constant; NeuroCombat cannot be applied")

    harmonized_features = features.copy()
    combat_dat = features[:, variable_feature_mask].T
    covars = pd.DataFrame({"site": site_ids.astype(str)})
    result = neuroCombat(dat=combat_dat, covars=covars, batch_col="site")
    harmonized_features[:, variable_feature_mask] = np.asarray(result["data"]).T

    harmonized_images = harmonized_features.reshape(n_images, 1, height, width).astype(np.float32)
    if not np.isfinite(harmonized_images).all():
        raise AssertionError("NeuroCombat output contains NaN or inf values")
    return harmonized_images


def save_manifest(
    path: Path,
    subject_ids: np.ndarray,
    site_ids: np.ndarray,
    slice_indices: np.ndarray,
    split: str,
) -> None:
    manifest = pd.DataFrame(
        {
            "row_idx": np.arange(len(subject_ids), dtype=np.int64),
            "subject_id": subject_ids,
            "site_id": site_ids,
            "slice_idx": slice_indices,
            "method": METHOD,
            "split": split,
        }
    )
    manifest.to_csv(path, index=False)


def save_qc_grid(
    raw_images: np.ndarray,
    harmonized_images: np.ndarray,
    subject_ids: np.ndarray,
    site_ids: np.ndarray,
    path: Path,
    max_subjects: int = 6,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional debug artifact
        print(f"Skipping QC image because matplotlib could not be imported: {exc}")
        return

    n = min(max_subjects, raw_images.shape[0])
    if n == 0:
        return

    fig, axes = plt.subplots(2, n, figsize=(2.4 * n, 4.8), squeeze=False)
    for col in range(n):
        raw = raw_images[col, 0]
        harmonized = harmonized_images[col, 0]
        vmin = float(min(raw.min(), harmonized.min()))
        vmax = float(max(raw.max(), harmonized.max()))

        axes[0, col].imshow(raw, cmap="gray", vmin=vmin, vmax=vmax)
        axes[0, col].set_title(f"{subject_ids[col]}\nsite {site_ids[col]}", fontsize=8)
        axes[0, col].axis("off")

        axes[1, col].imshow(harmonized, cmap="gray", vmin=vmin, vmax=vmax)
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("raw", fontsize=10)
    axes[1, 0].set_ylabel(METHOD, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        out_hw = parse_out_hw(args.out_hw)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    set_seed(args.seed)
    import_neurocombat()

    output_dir = Path(args.out_dir) / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(args, out_hw)
    raw_images, subject_ids, site_ids, slice_indices = extract_fixed_slices(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        out_hw=out_hw,
    )

    print(f"Split: {args.split}")
    print(f"Subjects: {raw_images.shape[0]}")
    print(f"Unique sites: {len(np.unique(site_ids))} -> {np.unique(site_ids).tolist()}")
    print("First deterministic rows:")
    for row in range(min(5, len(subject_ids))):
        print(
            f"  {row}: subject_id={subject_ids[row]} "
            f"site_id={int(site_ids[row])} slice_idx={int(slice_indices[row])}"
        )
    describe_array("Before NeuroCombat", raw_images)

    harmonized_images = apply_neurocombat_to_pixels(raw_images, site_ids)

    if harmonized_images.shape[0] != len(dataset):
        raise AssertionError(
            f"Output N={harmonized_images.shape[0]} does not match dataset N={len(dataset)}"
        )
    if harmonized_images.shape[1:] != (1, out_hw[0], out_hw[1]):
        raise AssertionError(
            f"Expected output image shape {(1, out_hw[0], out_hw[1])}, "
            f"got {harmonized_images.shape[1:]}"
        )
    if not np.isfinite(harmonized_images).all():
        raise AssertionError("Harmonized outputs contain NaN or inf values")

    describe_array("After NeuroCombat", harmonized_images)

    npz_path = output_dir / "neurocombat_slices.npz"
    np.savez_compressed(
        npz_path,
        images=harmonized_images,
        raw_images=raw_images,
        subject_ids=subject_ids,
        site_ids=site_ids,
        slice_indices=slice_indices,
        split=np.asarray(args.split),
        method=np.asarray(METHOD),
    )

    manifest_path = output_dir / "manifest.csv"
    save_manifest(
        manifest_path,
        subject_ids=subject_ids,
        site_ids=site_ids,
        slice_indices=slice_indices,
        split=args.split,
    )

    if not args.no_qc:
        save_qc_grid(
            raw_images=raw_images,
            harmonized_images=harmonized_images,
            subject_ids=subject_ids,
            site_ids=site_ids,
            path=output_dir / "qc_raw_vs_neurocombat.png",
        )

    config = {
        "method": METHOD,
        "split": args.split,
        "manifest_path": args.manifest_path,
        "splits_path": args.splits_path,
        "out_hw": list(out_hw),
        "slice_mode": "fixed",
        "input_mode": "image",
        "mask_mode": "none",
        "bg_suppress": True,
        "valid_nonzero_frac": args.valid_nonzero_frac,
        "fg_bbox_thr": args.fg_bbox_thr,
        "seed": args.seed,
        "n_subjects": int(raw_images.shape[0]),
        "n_sites": int(len(np.unique(site_ids))),
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    print(f"Saved: {npz_path}")
    print(f"Saved: {manifest_path}")


if __name__ == "__main__":
    main()
