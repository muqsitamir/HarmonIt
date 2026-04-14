import numpy as np

from harmonit.metrics import (
    compute_classification_metrics,
    site_classification_accuracy,
    compute_ssim,
    compute_psnr,
    feature_similarity,
    kl_divergence,
    wasserstein_dist,
)


def evaluate_all_metrics(
    y_true=None,
    y_pred=None,
    y_prob=None,
    y_true_site=None,
    y_pred_site=None,
    img1=None,
    img2=None,
    feat1=None,
    feat2=None,
    dist1=None,
    dist2=None,
):
    results = {}

    # --- Classification ---
    if y_true is not None and y_pred is not None:
        results["classification"] = compute_classification_metrics(
            y_true, y_pred, y_prob
        )

    # --- Site leakage ---
    if y_true_site is not None and y_pred_site is not None:
        results["site_leakage"] = site_classification_accuracy(
            y_true_site, y_pred_site
        )

    # --- Image quality ---
    if img1 is not None and img2 is not None:
        results["ssim"] = compute_ssim(img1, img2)
        results["psnr"] = compute_psnr(img1, img2)

    # --- Feature similarity ---
    if feat1 is not None and feat2 is not None:
        results["feature_similarity"] = feature_similarity(feat1, feat2)

    # --- Distribution alignment ---
    if dist1 is not None and dist2 is not None:
        results["kl_divergence"] = kl_divergence(dist1, dist2)
        results["wasserstein_distance"] = wasserstein_dist(dist1, dist2)

    return results