from pathlib import Path
import os
import numpy as np

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from harmonit.data.abide_slices_dataset import AbideSlicesDataset


def main():
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    split = os.getenv("SPLIT", "train")
    mask_mode = os.getenv("MASK_MODE", "none")  # none | bg_only | brain_only
    out_hw = tuple(map(int, os.getenv("OUT_HW", "256,256").split(",")))
    batch_size = int(os.getenv("BATCH_SIZE", "16"))
    num_workers = int(os.getenv("NUM_WORKERS", "0"))
    n_batches = int(os.getenv("N_BATCHES", "1"))
    valid_fg_frac = float(os.getenv("VALID_FG_FRAC", "0.02"))
    input_mode = os.getenv("INPUT_MODE", "image")  # image | mask_only
    bbox_jitter = int(os.getenv("BBOX_JITTER", "0"))

    bg_suppress = os.getenv("BG_SUPPRESS", "1") == "1"
    head_mask_thr = float(os.getenv("HEAD_MASK_THR", "0.08"))
    head_mask_dilate = int(os.getenv("HEAD_MASK_DILATE", "3"))

    # Fixed display range for imshow (helps avoid misleading auto-contrast)
    vmin = float(os.getenv("VMIN", "-2.0"))
    vmax = float(os.getenv("VMAX", "2.0"))

    ds = AbideSlicesDataset(
        manifest_path="data/abide_manifest.csv",
        splits_path="data/splits.json",
        split=split,
        out_hw=out_hw,
        slice_mode="random" if split == "train" else "fixed",
        valid_nonzero_frac=valid_fg_frac,
        seed=42,
        mask_mode=mask_mode,
        bg_suppress=bg_suppress,
        head_mask_thr=head_mask_thr,
        head_mask_dilate=head_mask_dilate,
        input_mode=input_mode,
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
    )

    for b_idx, batch in enumerate(loader, start=1):
        imgs, site_ids, subject_ids, slice_idxs = batch

        print("\n--- QC Batch", b_idx, "---")
        print(
            "Split:",
            split,
            "| mask_mode:",
            mask_mode,
            "| out_hw:",
            out_hw,
            "| bg_suppress:",
            bg_suppress,
            "| head_mask_thr:",
            head_mask_thr,
            "| head_mask_dilate:",
            head_mask_dilate,
        )
        print("Batch shapes:", imgs.shape)  # [B,1,H,W]
        print("Site IDs:", site_ids.tolist())
        print("Subjects:", list(subject_ids))
        print("Slice idx:", slice_idxs.tolist())

        # quick numeric sanity stats
        x = imgs.numpy()
        print("Value stats: min", float(x.min()), "max", float(x.max()), "mean", float(x.mean()), "std", float(x.std()))
        nz_frac = float((np.abs(x) > 1e-6).mean())
        print("Nonzero frac:", nz_frac)

        B = imgs.shape[0]
        ncols = 4
        nrows = (B + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows))
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

        for i in range(nrows * ncols):
            ax = axes[i]
            ax.axis("off")
            if i >= B:
                continue
            im = imgs[i, 0].numpy()
            ax.imshow(im.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)

            ax.set_title(f"{subject_ids[i]} | site={int(site_ids[i])} | z={int(slice_idxs[i])}", fontsize=8)

        fig.tight_layout()
        out_path = out_dir / f"qc_MM-{mask_mode}_IM-{input_mode}_BS-{bg_suppress}.png"
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        print(f"Saved QC grid -> {out_path}")

        if b_idx >= n_batches:
            break


if __name__ == "__main__":
    main()