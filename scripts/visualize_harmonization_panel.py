"""Create a paper-style visual comparison panel for harmonized NPZ outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_method(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use LABEL=PATH for each --method entry")
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("Method label cannot be empty")
    return label, Path(path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a single-subject harmonization comparison panel.")
    parser.add_argument("--method", action="append", type=parse_method, required=True, help="LABEL=NPZ path")
    parser.add_argument("--out", required=True, help="Output image path, e.g. outputs/figures/panel.png")
    parser.add_argument("--subject-id", default=None, help="Specific subject_id. If omitted, choose a representative non-target subject.")
    parser.add_argument("--target-site-id", type=int, default=5, help="Target/reference site id, default NYU=5.")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--format", choices=("png", "pdf", "both"), default="both")
    parser.add_argument("--ncols", type=int, default=0, help="Number of panel columns. Default uses one row.")
    parser.add_argument("--title", default=None)
    return parser


def load_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def as_str_array(values: np.ndarray) -> np.ndarray:
    return np.asarray(values).astype(str)


def choose_subject(reference: dict[str, Any], target_site_id: int) -> str:
    subjects = as_str_array(reference["subject_ids"])
    sites = np.asarray(reference["site_ids"], dtype=int)
    raw = np.asarray(reference["raw_images"], dtype=np.float32)

    candidates = np.flatnonzero(sites != target_site_id)
    if candidates.size == 0:
        candidates = np.arange(len(subjects))

    # Prefer a visually useful slice: substantial foreground, but not an extreme.
    foreground = (raw[:, 0] > 0.03).mean(axis=(1, 2))
    candidate_scores = foreground[candidates]
    target = np.quantile(candidate_scores, 0.70)
    chosen = candidates[int(np.argmin(np.abs(candidate_scores - target)))]
    return str(subjects[chosen])


def row_for_subject(artifact: dict[str, Any], subject_id: str) -> int:
    subjects = as_str_array(artifact["subject_ids"])
    hits = np.flatnonzero(subjects == subject_id)
    if hits.size == 0:
        raise ValueError(f"Subject {subject_id} not found in artifact")
    return int(hits[0])


def robust_window(images: list[np.ndarray]) -> tuple[float, float]:
    stacked = np.concatenate([img.reshape(-1) for img in images])
    foreground = stacked[stacked > 0.01]
    values = foreground if foreground.size else stacked
    lo, hi = np.percentile(values, [1, 99.5])
    if hi <= lo:
        return 0.0, 1.0
    return float(lo), float(hi)


def save_panel(
    out_path: Path,
    method_entries: list[tuple[str, Path]],
    subject_id: str | None,
    target_site_id: int,
    title: str | None,
    dpi: int,
    ncols: int,
) -> None:
    artifacts = [(label, load_npz(path), path) for label, path in method_entries]
    reference = artifacts[0][1]
    if "raw_images" not in reference:
        raise KeyError(f"Reference artifact lacks raw_images: {artifacts[0][2]}")

    subject_id = subject_id or choose_subject(reference, target_site_id=target_site_id)
    ref_row = row_for_subject(reference, subject_id)
    ref_subjects = as_str_array(reference["subject_ids"])
    ref_sites = np.asarray(reference["site_ids"], dtype=int)
    ref_slices = np.asarray(reference["slice_indices"], dtype=int)

    raw = np.asarray(reference["raw_images"], dtype=np.float32)[ref_row, 0]
    panels: list[tuple[str, np.ndarray]] = [("Raw input", raw)]

    for label, artifact, path in artifacts:
        row = row_for_subject(artifact, subject_id)
        if int(np.asarray(artifact["site_ids"], dtype=int)[row]) != int(ref_sites[ref_row]):
            raise ValueError(f"Site mismatch for {label}: {path}")
        if int(np.asarray(artifact["slice_indices"], dtype=int)[row]) != int(ref_slices[ref_row]):
            raise ValueError(f"Slice mismatch for {label}: {path}")
        panels.append((label, np.asarray(artifact["images"], dtype=np.float32)[row, 0]))

    vmin, vmax = robust_window([image for _label, image in panels])
    total = len(panels)
    ncols = int(ncols) if int(ncols) > 0 else total
    ncols = max(1, min(ncols, total))
    nrows = int(np.ceil(total / ncols))
    fig_w = max(7.0, 1.75 * ncols)
    fig_h = max(3.0, 1.85 * nrows + 0.35)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), constrained_layout=True)
    axes_flat = np.asarray(axes).reshape(-1)

    for ax, (label, image) in zip(axes_flat, panels):
        ax.imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(label, fontsize=8, pad=5)
        ax.set_axis_off()
    for ax in axes_flat[len(panels) :]:
        ax.set_axis_off()

    site_id = int(ref_sites[ref_row])
    slice_idx = int(ref_slices[ref_row])
    suptitle = title or f"Held-out subject {subject_id} | site_id={site_id} | axial slice={slice_idx}"
    fig.suptitle(suptitle, fontsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"saved={out_path}")
    print(f"subject_id={subject_id} site_id={site_id} slice_idx={slice_idx}")


def main() -> None:
    args = build_arg_parser().parse_args()
    out_path = Path(args.out)
    stems: list[Path]
    if args.format == "both":
        stems = [out_path.with_suffix(".png"), out_path.with_suffix(".pdf")]
    elif args.format == "png":
        stems = [out_path.with_suffix(".png")]
    else:
        stems = [out_path.with_suffix(".pdf")]

    subject_id = args.subject_id
    for path in stems:
        save_panel(
            out_path=path,
            method_entries=args.method,
            subject_id=subject_id,
            target_site_id=args.target_site_id,
            title=args.title,
            dpi=args.dpi,
            ncols=args.ncols,
        )


if __name__ == "__main__":
    main()
