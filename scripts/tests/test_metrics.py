from torch.utils.data import DataLoader

from harmonit.data.abide_slices_dataset import AbideSlicesDataset

# Import metrics
from harmonit.metrics.structural_preservation import compute_psnr
from harmonit.metrics.feature_consistency import feature_similarity, cross_correlation
from harmonit.metrics.distribution_alignment import kl_divergence, wasserstein_dist


def main():
    print("[Testing metrics on ABIDE data]")

    dataset = AbideSlicesDataset(
        manifest_path="data/abide_manifest.csv",
        splits_path="data/splits.json",
        split="val",
        out_hw=(256, 256),
        slice_mode="fixed",
        seed=42,
    )

    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    # Get one batch
    batch = next(iter(loader))
    x, y, subject_ids, _ = batch

    img1 = x[0, 0].numpy()
    img2 = x[1, 0].numpy()

    print("\n--- Structural metrics ---")
    print("PSNR:", compute_psnr(img1, img2))

    print("\n--- Feature metrics ---")
    feat1 = img1.flatten()
    feat2 = img2.flatten()

    print("Feature similarity:", feature_similarity(feat1, feat2))
    print("Cross-correlation:", cross_correlation(feat1, feat2))

    print("\n--- Distribution metrics ---")
    dist1 = img1.flatten()
    dist2 = img2.flatten()

    print("KL divergence:", kl_divergence(dist1, dist1))
    print("Wasserstein:", wasserstein_dist(dist1, dist2))


if __name__ == "__main__":
    main()