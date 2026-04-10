

import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from harmonit.data.abide_slices_dataset import AbideSlicesDataset, make_head_mask


def main():
    # ---- Config (env overridable) ----
    split = os.getenv("SPLIT", "val")  # train|val|test
    n_samples = int(os.getenv("N_SAMPLES", "40"))
    out_hw = tuple(map(int, os.getenv("OUT_HW", "256,256").split(",")))

    # Mask params must match the dataset defaults (and be logged for reproducibility)
    head_mask_thr = float(os.getenv("HEAD_MASK_THR", "0.08"))
    head_mask_dilate = int(os.getenv("HEAD_MASK_DILATE", "3"))

    # Dataset / preprocessing params
    valid_fg_frac = float(os.getenv("VALID_FG_FRAC", "0.02"))
    volume_cache_size = int(os.getenv("VOLUME_CACHE_SIZE", "8"))

    # We want to QC the mask itself, so do NOT suppress background inside the dataset.
    bg_suppress = False
    mask_mode = "none"

    # Output
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(os.getenv("OUT_DIR", f"results/mask_qc_{split}_{stamp}"))
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    print("Mask QC")
    print(" split:", split)
    print(" n_samples:", n_samples)
    print(" out_hw:", out_hw)
    print(" head_mask_thr:", head_mask_thr)
    print(" head_mask_dilate:", head_mask_dilate)
    print(" valid_fg_frac:", valid_fg_frac)
    print(" out_dir:", out_dir)

    # ---- Load dataset exactly as the probe sees it (except bg_suppress=False for QC) ----
    ds = AbideSlicesDataset(
        manifest_path="data/abide_manifest.csv",
        splits_path="data/splits.json",
        split=split,
        out_hw=out_hw,
        slice_mode="random" if split == "train" else "fixed",
        valid_nonzero_frac=valid_fg_frac,
        seed=123,
        volume_cache_size=volume_cache_size,
        mask_mode=mask_mode,
        bg_suppress=bg_suppress,
        head_mask_thr=head_mask_thr,
        head_mask_dilate=head_mask_dilate,
    )

    # ---- Balanced sampling across sites (stress test across domains) ----
    # Build a map: site -> list of dataset indices
    site_to_indices = {}
    for i, s in enumerate(ds.samples):
        site_to_indices.setdefault(s.site, []).append(i)

    sites = sorted(site_to_indices.keys())
    n_sites = len(sites)
    per_site = max(1, int(np.ceil(n_samples / max(1, n_sites))))

    chosen = []
    rng = np.random.RandomState(0)
    for site in sites:
        idxs = site_to_indices[site]
        take = min(per_site, len(idxs))
        chosen.extend(rng.choice(idxs, size=take, replace=False).tolist())

    # If we overshot, truncate to n_samples
    chosen = chosen[:n_samples]
    print(f"Sampling {len(chosen)} examples across {n_sites} sites (per_site={per_site}).")

    rows = []
    flagged = []

    for j, idx in enumerate(chosen, start=1):
        img_t, site_id, subject_id, z = ds[idx]
        img = img_t[0].numpy().astype(np.float32)  # [H,W]

        mask = make_head_mask(img, thr=head_mask_thr, dilate_iters=head_mask_dilate)
        masked = img.copy()
        masked[~mask] = 0.0

        # Stats: how much area and energy are retained
        area_frac = float(mask.mean())
        denom = float(np.sum(np.abs(img))) + 1e-8
        retained_energy = float(np.sum(np.abs(img) * mask) / denom)

        # Quick heuristics to flag potential failures (too small/too big)
        if area_frac < 0.15 or area_frac > 0.95:
            flagged.append((subject_id, int(site_id), int(z), area_frac, retained_energy))

        # Save a 3-panel QC figure
        fig = plt.figure(figsize=(12, 4))
        ax1 = plt.subplot(1, 3, 1)
        ax2 = plt.subplot(1, 3, 2)
        ax3 = plt.subplot(1, 3, 3)

        ax1.imshow(img.T, cmap="gray", origin="lower")
        ax1.set_title("Input (normalized)")
        ax1.axis("off")

        ax2.imshow(img.T, cmap="gray", origin="lower")
        # Overlay mask contour
        try:
            ax2.contour(mask.T.astype(float), levels=[0.5], linewidths=1.0)
        except Exception:
            pass
        ax2.set_title("Head mask overlay")
        ax2.axis("off")

        ax3.imshow(masked.T, cmap="gray", origin="lower")
        ax3.set_title("Masked (bg=0)")
        ax3.axis("off")

        fig.suptitle(f"{subject_id} | site_id={int(site_id)} | z={int(z)} | area={area_frac:.3f} | energy={retained_energy:.3f}", fontsize=10)
        fig.tight_layout(rect=[0, 0.02, 1, 0.92])

        out_path = img_dir / f"{j:03d}_{subject_id}_site{int(site_id)}_z{int(z)}.png"
        fig.savefig(out_path, dpi=160)
        plt.close(fig)

        rows.append(
            {
                "idx": int(idx),
                "subject_id": subject_id,
                "site_id": int(site_id),
                "z": int(z),
                "mask_area_frac": area_frac,
                "retained_energy_frac": retained_energy,
                "qc_path": str(out_path),
            }
        )

        if j % 10 == 0 or j == 1:
            print(f"[{j}/{len(chosen)}] saved {out_path.name} | area={area_frac:.3f} energy={retained_energy:.3f}")

    df = pd.DataFrame(rows)
    csv_path = out_dir / "mask_qc_report.csv"
    df.to_csv(csv_path, index=False)
    print("Wrote CSV ->", csv_path)

    # Summary stats
    summary = {
        "n": int(len(df)),
        "mask_area_mean": float(df["mask_area_frac"].mean()),
        "mask_area_std": float(df["mask_area_frac"].std()),
        "energy_mean": float(df["retained_energy_frac"].mean()),
        "energy_std": float(df["retained_energy_frac"].std()),
        "n_flagged": int(len(flagged)),
    }
    (out_dir / "mask_qc_summary.json").write_text(pd.Series(summary).to_json(indent=2))

    print("Summary:")
    for k, v in summary.items():
        print(" ", k, ":", v)

    if flagged:
        print("\nFlagged (potential failures: very small/large mask area):")
        for subj, sid, z, a, e in flagged[:20]:
            print(f"  {subj} | site_id={sid} | z={z} | area={a:.3f} | energy={e:.3f}")
        if len(flagged) > 20:
            print(f"  ... and {len(flagged) - 20} more")


if __name__ == "__main__":
    main()