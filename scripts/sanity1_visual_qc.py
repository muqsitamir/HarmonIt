import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


def percentile_window(x: np.ndarray, p_low=2, p_high=98):
    """Robust intensity window so the brain looks good even across scanners."""
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(x, [p_low, p_high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(x.min()), float(x.max())
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return 0.0, 1.0
    return float(lo), float(hi)


def make_qc_figure(nifti_path: Path, out_png: Path):
    img = nib.load(str(nifti_path))
    data = img.get_fdata(dtype=np.float32)

    # Allow (X,Y,Z,1) and squeeze it into 3D
    if data.ndim == 4 and data.shape[3] == 1:
        data = np.squeeze(data, axis=3)

    if data.ndim != 3:
        raise ValueError(f"Expected 3D (or 3D+1). Got shape={data.shape}")

    shape = data.shape
    zooms = img.header.get_zooms()[:3]

    # Print metadata (this is what you’ll screenshot for “metadata proof”)
    print("=== Sanity Check 1 Visual QC ===")
    print(f"File:   {nifti_path}")
    print(f"Shape:  {shape}")
    print(f"Zooms:  {tuple(np.round(zooms, 4))} mm")
    print(f"Dtype:  {img.header.get_data_dtype()}")

    # Choose center slices
    cx, cy, cz = (shape[0] // 2, shape[1] // 2, shape[2] // 2)

    # Transpose for display so axes look natural in imshow
    sag = data[cx, :, :].T      # x fixed
    cor = data[:, cy, :].T      # y fixed
    axi = data[:, :, cz].T      # z fixed

    vmin, vmax = percentile_window(data)

    fig = plt.figure(figsize=(12, 4.6), dpi=200)
    gs = fig.add_gridspec(1, 3, wspace=0.05)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    for ax, slc, title in zip(
        axes,
        [sag, cor, axi],
        [f"Sagittal (x={cx})", f"Coronal (y={cy})", f"Axial (z={cz})"],
    ):
        ax.imshow(slc, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    meta = (
        f"File: {nifti_path.name}\n"
        f"Shape: {shape}\n"
        f"Voxel spacing (mm): {tuple(np.round(zooms, 4))}\n"
        f"Window (p2–p98): {vmin:.2f} – {vmax:.2f}"
    )
    fig.text(0.01, -0.02, meta, fontsize=9, family="monospace", va="top")
    fig.tight_layout(rect=[0, 0.08, 1, 1])

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {out_png}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nifti", required=True, help="Path to a .nii or .nii.gz file")
    parser.add_argument("--out", default="figures/sanity1_visual_check.png", help="Output PNG path")
    args = parser.parse_args()

    make_qc_figure(Path(args.nifti), Path(args.out))


if __name__ == "__main__":
    main()