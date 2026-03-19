from pathlib import Path

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from harmonit.data.abide_slices_dataset import AbideSlicesDataset


def main():
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = AbideSlicesDataset(
        manifest_path="data/abide_manifest.csv",
        splits_path="data/splits.json",
        split="train",
        out_hw=(256, 256),
        slice_mode="random",
        valid_nonzero_frac=0.02,
        seed=42,
    )

    loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)

    batch = next(iter(loader))
    imgs, site_ids, subject_ids, slice_idxs = batch

    print("Batch shapes:", imgs.shape)  # [B,1,H,W]
    print("Site IDs:", site_ids.tolist())
    print("Subjects:", list(subject_ids))
    print("Slice idx:", slice_idxs.tolist())

    # make grid
    B = imgs.shape[0]
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()

    for i in range(B):
        ax = axes[i]
        im = imgs[i, 0].numpy()
        ax.imshow(im.T, cmap="gray", origin="lower")
        ax.set_title(f"{subject_ids[i]} | site={int(site_ids[i])} | z={int(slice_idxs[i])}", fontsize=8)
        ax.axis("off")

    fig.tight_layout()
    out_path = out_dir / "qc_dataloader_batch.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    print(f"Saved QC grid -> {out_path}")


if __name__ == "__main__":
    main()