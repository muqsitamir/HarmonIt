"""Evaluate a harmonized fixed-slice NPZ against raw ABIDE deterministic slices.

The input NPZ is expected to follow the artifact format produced by
``scripts/methods/neurocombat.py``:

  - images: harmonized images, shape [N, 1, H, W]
  - raw_images: optional raw fixed slices, shape [N, 1, H, W]
  - subject_ids
  - site_ids
  - slice_indices
  - split
  - method

This evaluator reloads ``AbideSlicesDataset`` with ``slice_mode="fixed"``,
``input_mode="image"``, ``mask_mode="none"``, and ``bg_suppress=True`` so the
metric reference is the same deterministic raw subject-slice policy used by the
site-probe benchmark. It does not tune harmonization parameters, train a model,
or modify the frozen site probe.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from harmonit.metrics.distribution_alignment import kl_divergence, wasserstein_dist
from harmonit.metrics.feature_consistency import cross_correlation, feature_similarity
from harmonit.metrics.structural_preservation import compute_psnr
from harmonit.utils.metrics import confusion_and_balanced_acc


def parse_out_hw(values: Iterable[int]) -> tuple[int, int]:
    values = tuple(int(v) for v in values)
    if len(values) == 1:
        return values[0], values[0]
    if len(values) == 2:
        return values[0], values[1]
    raise argparse.ArgumentTypeError("--out-hw expects one integer or two integers")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate harmonized NPZ outputs against fixed raw ABIDE slices."
    )
    parser.add_argument(
        "--npz-path",
        required=True,
        help="Path to harmonized NPZ, e.g. outputs/harmonized/neurocombat/test/neurocombat_slices.npz.",
    )
    parser.add_argument("--manifest-path", default="data/abide_manifest.csv")
    parser.add_argument("--splits-path", default="data/splits.json")
    parser.add_argument(
        "--split",
        default=None,
        choices=("train", "val", "test"),
        help="Defaults to the NPZ split field when present, otherwise test.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Defaults to <npz parent>/metrics.",
    )
    parser.add_argument("--out-hw", nargs="+", type=int, default=(256, 256))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--valid-nonzero-frac",
        type=float,
        default=float(os.getenv("VALID_FG_FRAC", "0.02")),
        help="Must match fixed-slice site-probe evaluation settings.",
    )
    parser.add_argument(
        "--fg-bbox-thr",
        type=float,
        default=float(os.getenv("FG_BBOX_THR", "0.02")),
        help="Must match fixed-slice site-probe evaluation settings.",
    )
    parser.add_argument("--volume-cache-size", type=int, default=12)
    parser.add_argument("--distribution-bins", type=int, default=50)
    parser.add_argument(
        "--site-probe-ckpt",
        default="checkpoints/site_probe/model_best.pt",
        help="Frozen site-probe checkpoint. Site-probe metrics are skipped if missing.",
    )
    parser.add_argument(
        "--skip-site-probe",
        action="store_true",
        help="Only compute preservation and distribution metrics.",
    )
    return parser


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def scalar_from_npz(value: Any) -> str | None:
    if value is None:
        return None
    array = np.asarray(value)
    if array.shape == ():
        return str(array.item())
    if array.size == 1:
        return str(array.reshape(-1)[0])
    return None


def load_npz_artifact(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing NPZ artifact: {path}")

    with np.load(path, allow_pickle=False) as data:
        required = {"images", "subject_ids", "site_ids", "slice_indices"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise KeyError(f"NPZ is missing required arrays: {missing}")

        artifact = {key: data[key] for key in data.files}

    images = np.asarray(artifact["images"], dtype=np.float32)
    if images.ndim != 4 or images.shape[1] != 1:
        raise ValueError(f"Expected images shape [N, 1, H, W], got {images.shape}")
    if not np.isfinite(images).all():
        raise ValueError("Harmonized images contain NaN or inf values")

    artifact["images"] = images
    artifact["subject_ids"] = np.asarray(artifact["subject_ids"]).astype(str)
    artifact["site_ids"] = np.asarray(artifact["site_ids"], dtype=np.int64)
    artifact["slice_indices"] = np.asarray(artifact["slice_indices"], dtype=np.int64)
    artifact["split"] = scalar_from_npz(artifact.get("split"))
    artifact["method"] = scalar_from_npz(artifact.get("method")) or "unknown"

    n = images.shape[0]
    for key in ("subject_ids", "site_ids", "slice_indices"):
        if len(artifact[key]) != n:
            raise ValueError(f"{key} length {len(artifact[key])} does not match images N={n}")

    if "raw_images" in artifact:
        raw_images = np.asarray(artifact["raw_images"], dtype=np.float32)
        if raw_images.shape != images.shape:
            raise ValueError(
                f"raw_images shape {raw_images.shape} does not match images shape {images.shape}"
            )
        if not np.isfinite(raw_images).all():
            raise ValueError("NPZ raw_images contain NaN or inf values")
        artifact["raw_images"] = raw_images

    return artifact


def build_raw_dataset(args: argparse.Namespace, split: str, out_hw: tuple[int, int]) -> Any:
    try:
        from harmonit.data.abide_slices_dataset import AbideSlicesDataset
    except ImportError as exc:
        raise ImportError(
            "Could not import AbideSlicesDataset. Run from the repository root with "
            "the project runtime dependencies installed."
        ) from exc

    dataset = AbideSlicesDataset(
        manifest_path=args.manifest_path,
        splits_path=args.splits_path,
        split=split,
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
    dataset.aug_affine = False
    return dataset


def extract_raw_fixed_slices(
    dataset: Any,
    batch_size: int,
    num_workers: int,
    out_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    if len(dataset) == 0:
        raise ValueError("ABIDE split is empty; no raw slices to evaluate")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=(num_workers > 0),
    )

    raw_batches: list[np.ndarray] = []
    subject_ids: list[str] = []
    site_ids: list[int] = []
    slice_indices: list[int] = []

    for images, sites, subjects, slices in tqdm(loader, desc="Loading raw fixed ABIDE slices"):
        expected_shape = (images.shape[0], 1, out_hw[0], out_hw[1])
        if tuple(images.shape) != expected_shape:
            raise AssertionError(
                f"Expected raw batch image shape {expected_shape}, got {tuple(images.shape)}"
            )
        raw_batches.append(images.cpu().numpy().astype(np.float32, copy=False))
        subject_ids.extend(str(subject_id) for subject_id in subjects)
        site_ids.extend(int(site_id) for site_id in sites.cpu().numpy().tolist())
        slice_indices.extend(int(slice_idx) for slice_idx in slices.cpu().numpy().tolist())

    raw_images = np.concatenate(raw_batches, axis=0)
    if raw_images.shape[0] != len(dataset):
        raise AssertionError(
            f"Loaded {raw_images.shape[0]} raw images, but dataset has {len(dataset)} samples"
        )
    if not np.isfinite(raw_images).all():
        raise AssertionError("Raw fixed slices contain NaN or inf values")

    return (
        raw_images,
        np.asarray(subject_ids, dtype=str),
        np.asarray(site_ids, dtype=np.int64),
        np.asarray(slice_indices, dtype=np.int64),
    )


def assert_artifact_matches_raw(
    artifact: dict[str, Any],
    raw_images: np.ndarray,
    raw_subject_ids: np.ndarray,
    raw_site_ids: np.ndarray,
    raw_slice_indices: np.ndarray,
    out_hw: tuple[int, int],
) -> None:
    images = artifact["images"]
    if images.shape != raw_images.shape:
        raise ValueError(
            f"Harmonized images shape {images.shape} does not match raw fixed slices "
            f"shape {raw_images.shape}"
        )
    if images.shape[1:] != (1, out_hw[0], out_hw[1]):
        raise ValueError(f"Expected image shape [N, 1, {out_hw[0]}, {out_hw[1]}], got {images.shape}")

    checks = [
        ("subject_ids", artifact["subject_ids"], raw_subject_ids),
        ("site_ids", artifact["site_ids"], raw_site_ids),
        ("slice_indices", artifact["slice_indices"], raw_slice_indices),
    ]
    for name, got, expected in checks:
        if not np.array_equal(got, expected):
            mismatch = int(np.flatnonzero(got != expected)[0])
            raise ValueError(
                f"NPZ {name} does not match fixed raw dataset at row {mismatch}: "
                f"npz={got[mismatch]!r}, raw={expected[mismatch]!r}"
            )

    if "raw_images" in artifact and not np.allclose(artifact["raw_images"], raw_images, atol=1e-5):
        max_abs_diff = float(np.max(np.abs(artifact["raw_images"] - raw_images)))
        raise ValueError(
            "NPZ raw_images do not match freshly loaded deterministic raw slices "
            f"(max abs diff={max_abs_diff:.6g})"
        )


def summarize(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "median": float(np.median(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def compute_pairwise_metrics(
    raw_images: np.ndarray,
    harmonized_images: np.ndarray,
    subject_ids: np.ndarray,
    site_ids: np.ndarray,
    slice_indices: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for row_idx in range(raw_images.shape[0]):
        raw = raw_images[row_idx, 0]
        harmonized = harmonized_images[row_idx, 0]
        raw_flat = raw.reshape(-1)
        harmonized_flat = harmonized.reshape(-1)
        diff = harmonized_flat - raw_flat

        rows.append(
            {
                "row_idx": row_idx,
                "subject_id": subject_ids[row_idx],
                "site_id": int(site_ids[row_idx]),
                "slice_idx": int(slice_indices[row_idx]),
                "psnr": float(compute_psnr(raw, harmonized)),
                "feature_similarity": float(feature_similarity(raw_flat, harmonized_flat)),
                "cross_correlation": float(cross_correlation(raw_flat, harmonized_flat)),
                "mse": float(np.mean(diff * diff)),
                "mae": float(np.mean(np.abs(diff))),
                "raw_mean": float(np.mean(raw_flat)),
                "harmonized_mean": float(np.mean(harmonized_flat)),
                "raw_std": float(np.std(raw_flat)),
                "harmonized_std": float(np.std(harmonized_flat)),
            }
        )
    return pd.DataFrame(rows)


def compute_distribution_metrics(
    raw_images: np.ndarray,
    harmonized_images: np.ndarray,
    site_ids: np.ndarray,
    bins: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    raw_flat = raw_images.reshape(-1)
    harmonized_flat = harmonized_images.reshape(-1)

    global_metrics = {
        "kl_raw_to_harmonized": float(kl_divergence(raw_flat, harmonized_flat, bins=bins)),
        "kl_harmonized_to_raw": float(kl_divergence(harmonized_flat, raw_flat, bins=bins)),
        "wasserstein": float(wasserstein_dist(raw_flat, harmonized_flat)),
        "raw_mean": float(np.mean(raw_flat)),
        "harmonized_mean": float(np.mean(harmonized_flat)),
        "raw_std": float(np.std(raw_flat)),
        "harmonized_std": float(np.std(harmonized_flat)),
    }

    rows = []
    for site_id in sorted(np.unique(site_ids).tolist()):
        mask = site_ids == site_id
        raw_site = raw_images[mask].reshape(-1)
        harmonized_site = harmonized_images[mask].reshape(-1)
        rows.append(
            {
                "site_id": int(site_id),
                "n_subjects": int(mask.sum()),
                "kl_raw_to_harmonized": float(kl_divergence(raw_site, harmonized_site, bins=bins)),
                "kl_harmonized_to_raw": float(kl_divergence(harmonized_site, raw_site, bins=bins)),
                "wasserstein": float(wasserstein_dist(raw_site, harmonized_site)),
                "raw_mean": float(np.mean(raw_site)),
                "harmonized_mean": float(np.mean(harmonized_site)),
                "raw_std": float(np.std(raw_site)),
                "harmonized_std": float(np.std(harmonized_site)),
            }
        )

    return global_metrics, pd.DataFrame(rows)


def build_site_probe(num_classes: int, state: dict[str, Any]) -> Any:
    import torch.nn as nn
    from torchvision.models import resnet18

    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if any(key.startswith("module.") for key in state.keys()):
        state = {key.replace("module.", "", 1): value for key, value in state.items()}

    model.load_state_dict(state, strict=True)
    return model


def evaluate_site_probe(
    raw_images: np.ndarray,
    harmonized_images: np.ndarray,
    site_ids: np.ndarray,
    ckpt_path: Path,
    batch_size: int,
) -> dict[str, Any]:
    import torch

    if not ckpt_path.exists():
        return {"status": "skipped", "reason": f"missing checkpoint: {ckpt_path}"}

    try:
        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    fc_key = "fc.weight"
    if fc_key not in state and "module.fc.weight" in state:
        fc_key = "module.fc.weight"
    if fc_key not in state:
        raise KeyError("Could not infer site-probe class count from checkpoint fc.weight")

    num_classes = int(state[fc_key].shape[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_site_probe(num_classes=num_classes, state=state).to(device)
    model.eval()

    def predict(images: np.ndarray) -> np.ndarray:
        preds = []
        with torch.no_grad():
            for start in range(0, images.shape[0], batch_size):
                batch = torch.from_numpy(images[start : start + batch_size]).float().to(device)
                logits = model(batch)
                preds.append(torch.argmax(logits, dim=1).cpu().numpy())
        return np.concatenate(preds, axis=0)

    raw_pred = predict(raw_images)
    harmonized_pred = predict(harmonized_images)
    raw_cm, raw_acc, raw_bal_acc = confusion_and_balanced_acc(site_ids, raw_pred, num_classes)
    harmonized_cm, harmonized_acc, harmonized_bal_acc = confusion_and_balanced_acc(
        site_ids, harmonized_pred, num_classes
    )

    return {
        "status": "computed",
        "checkpoint": str(ckpt_path),
        "device": str(device),
        "num_classes": num_classes,
        "raw_accuracy": raw_acc,
        "raw_balanced_accuracy": raw_bal_acc,
        "harmonized_accuracy": harmonized_acc,
        "harmonized_balanced_accuracy": harmonized_bal_acc,
        "accuracy_drop": raw_acc - harmonized_acc,
        "balanced_accuracy_drop": raw_bal_acc - harmonized_bal_acc,
        "raw_confusion_matrix": raw_cm,
        "harmonized_confusion_matrix": harmonized_cm,
    }


def write_key_value_csv(path: Path, values: dict[str, Any]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in values.items():
            if isinstance(value, (dict, list, tuple, np.ndarray)):
                continue
            writer.writerow([key, value])


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        out_hw = parse_out_hw(args.out_hw)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    set_seed(args.seed)

    npz_path = Path(args.npz_path)
    artifact = load_npz_artifact(npz_path)
    split = args.split or artifact["split"] or "test"
    method = artifact["method"]
    output_dir = Path(args.out_dir) if args.out_dir else npz_path.parent / "metrics"

    dataset = build_raw_dataset(args, split=split, out_hw=out_hw)
    raw_images, subject_ids, site_ids, slice_indices = extract_raw_fixed_slices(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        out_hw=out_hw,
    )
    harmonized_images = artifact["images"]
    assert_artifact_matches_raw(
        artifact=artifact,
        raw_images=raw_images,
        raw_subject_ids=subject_ids,
        raw_site_ids=site_ids,
        raw_slice_indices=slice_indices,
        out_hw=out_hw,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Method: {method}")
    print(f"Split: {split}")
    print(f"Subjects: {raw_images.shape[0]}")
    print(f"Unique sites: {len(np.unique(site_ids))} -> {np.unique(site_ids).tolist()}")
    print("First deterministic rows:")
    for row in range(min(5, len(subject_ids))):
        print(
            f"  {row}: subject_id={subject_ids[row]} "
            f"site_id={int(site_ids[row])} slice_idx={int(slice_indices[row])}"
        )

    pairwise_df = compute_pairwise_metrics(
        raw_images=raw_images,
        harmonized_images=harmonized_images,
        subject_ids=subject_ids,
        site_ids=site_ids,
        slice_indices=slice_indices,
    )
    global_distribution, site_distribution_df = compute_distribution_metrics(
        raw_images=raw_images,
        harmonized_images=harmonized_images,
        site_ids=site_ids,
        bins=args.distribution_bins,
    )

    site_probe_metrics: dict[str, Any]
    if args.skip_site_probe:
        site_probe_metrics = {"status": "skipped", "reason": "--skip-site-probe"}
    else:
        site_probe_metrics = evaluate_site_probe(
            raw_images=raw_images,
            harmonized_images=harmonized_images,
            site_ids=site_ids,
            ckpt_path=Path(args.site_probe_ckpt),
            batch_size=args.batch_size,
        )

    summary = {
        "method": method,
        "split": split,
        "npz_path": str(npz_path),
        "n_subjects": int(raw_images.shape[0]),
        "n_sites": int(len(np.unique(site_ids))),
        "out_hw": list(out_hw),
        "preservation": {
            "psnr": summarize(pairwise_df["psnr"].to_numpy()),
            "feature_similarity": summarize(pairwise_df["feature_similarity"].to_numpy()),
            "cross_correlation": summarize(pairwise_df["cross_correlation"].to_numpy()),
            "mse": summarize(pairwise_df["mse"].to_numpy()),
            "mae": summarize(pairwise_df["mae"].to_numpy()),
        },
        "distribution": global_distribution,
        "site_probe": {
            key: value
            for key, value in site_probe_metrics.items()
            if key not in {"raw_confusion_matrix", "harmonized_confusion_matrix"}
        },
    }

    pairwise_path = output_dir / "pairwise_preservation_metrics.csv"
    site_distribution_path = output_dir / "distribution_by_site.csv"
    summary_path = output_dir / "summary_metrics.json"
    summary_csv_path = output_dir / "summary_metrics.csv"

    pairwise_df.to_csv(pairwise_path, index=False)
    site_distribution_df.to_csv(site_distribution_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    flat_summary = {
        "method": method,
        "split": split,
        "n_subjects": int(raw_images.shape[0]),
        "n_sites": int(len(np.unique(site_ids))),
        "psnr_mean": summary["preservation"]["psnr"]["mean"],
        "feature_similarity_mean": summary["preservation"]["feature_similarity"]["mean"],
        "cross_correlation_mean": summary["preservation"]["cross_correlation"]["mean"],
        "mse_mean": summary["preservation"]["mse"]["mean"],
        "mae_mean": summary["preservation"]["mae"]["mean"],
        "global_kl_raw_to_harmonized": global_distribution["kl_raw_to_harmonized"],
        "global_kl_harmonized_to_raw": global_distribution["kl_harmonized_to_raw"],
        "global_wasserstein": global_distribution["wasserstein"],
        "site_probe_status": summary["site_probe"].get("status"),
        "raw_site_probe_accuracy": summary["site_probe"].get("raw_accuracy"),
        "harmonized_site_probe_accuracy": summary["site_probe"].get("harmonized_accuracy"),
        "site_probe_accuracy_drop": summary["site_probe"].get("accuracy_drop"),
        "raw_site_probe_balanced_accuracy": summary["site_probe"].get("raw_balanced_accuracy"),
        "harmonized_site_probe_balanced_accuracy": summary["site_probe"].get(
            "harmonized_balanced_accuracy"
        ),
        "site_probe_balanced_accuracy_drop": summary["site_probe"].get("balanced_accuracy_drop"),
    }
    write_key_value_csv(summary_csv_path, flat_summary)

    if "raw_confusion_matrix" in site_probe_metrics:
        np.save(output_dir / "site_probe_raw_cm.npy", site_probe_metrics["raw_confusion_matrix"])
        np.save(
            output_dir / "site_probe_harmonized_cm.npy",
            site_probe_metrics["harmonized_confusion_matrix"],
        )

    print("Summary:")
    print(f"  PSNR mean: {flat_summary['psnr_mean']:.6f}")
    print(f"  Feature similarity mean: {flat_summary['feature_similarity_mean']:.6f}")
    print(f"  Cross-correlation mean: {flat_summary['cross_correlation_mean']:.6f}")
    print(f"  Global KL raw->harmonized: {flat_summary['global_kl_raw_to_harmonized']:.6f}")
    print(f"  Global Wasserstein: {flat_summary['global_wasserstein']:.6f}")
    if summary["site_probe"].get("status") == "computed":
        print(
            "  Site-probe accuracy: "
            f"raw={flat_summary['raw_site_probe_accuracy']:.6f}, "
            f"harmonized={flat_summary['harmonized_site_probe_accuracy']:.6f}, "
            f"drop={flat_summary['site_probe_accuracy_drop']:.6f}"
        )
    else:
        print(f"  Site-probe: {summary['site_probe'].get('status')} ({summary['site_probe'].get('reason')})")

    print(f"Saved: {pairwise_path}")
    print(f"Saved: {site_distribution_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {summary_csv_path}")


if __name__ == "__main__":
    main()
