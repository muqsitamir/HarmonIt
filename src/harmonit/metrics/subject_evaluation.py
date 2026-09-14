"""Fixed ISBI intensity metrics and paired, site-stratified subject inference."""

import numpy as np
from scipy.stats import entropy, wasserstein_distance


METRICS = ("psnr", "mse", "mae", "pixel_cosine_similarity", "cross_correlation",
           "subject_wasserstein_raw_harm", "subject_kl_raw_harm")
HISTOGRAM_EDGES = np.r_[-np.inf, np.linspace(0.0, 1.0, 51), np.inf]


def probability_histogram(values):
    """50 fixed [0,1] bins plus explicit underflow/overflow bins; no clipping."""
    values = np.asarray(values, dtype=np.float64).ravel()
    if not values.size or not np.isfinite(values).all():
        raise ValueError("Histogram requires nonempty finite intensities")
    # Include 1.0 in the last central bin, reserving overflow for values > 1.
    edges = HISTOGRAM_EDGES.copy()
    edges[-2] = np.nextafter(1.0, np.inf)
    counts = np.histogram(values, bins=edges)[0].astype(np.float64)
    probabilities = counts / counts.sum()
    return (probabilities + 1e-8) / (1 + len(probabilities) * 1e-8)


def pixel_metrics(raw, harmonized):
    a, b = np.asarray(raw, dtype=np.float64), np.asarray(harmonized, dtype=np.float64)
    if a.shape != b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Images must be finite, nonempty and shape-matched")
    a, b = a.ravel(), b.ravel()
    delta = b - a
    mse = float(np.mean(delta ** 2))
    cosine_denom = np.linalg.norm(a) * np.linalg.norm(b)
    ac, bc = a - a.mean(), b - b.mean()
    corr_denom = np.linalg.norm(ac) * np.linalg.norm(bc)
    return {
        "psnr": float(-10 * np.log10(mse)) if mse > 0 else float("inf"),
        "mse": mse,
        "mae": float(np.abs(delta).mean()),
        "pixel_cosine_similarity": float(np.clip(np.dot(a, b) / cosine_denom, -1, 1)) if cosine_denom else float("nan"),
        "cross_correlation": float(np.clip(np.dot(ac, bc) / corr_denom, -1, 1)) if corr_denom else float("nan"),
        "subject_wasserstein_raw_harm": float(wasserstein_distance(a, b)),
        "subject_kl_raw_harm": float(entropy(probability_histogram(a), probability_histogram(b))),
        "exact_identity": bool(mse == 0),
        "near_identity": bool(np.max(np.abs(delta)) <= 1e-6),
        "output_below_zero_fraction": float((b < 0).mean()),
        "output_above_one_fraction": float((b > 1).mean()),
    }


def stratified_indices(site_ids, replicates=2000, seed=20260913):
    """Each replicate samples subjects within site and retains every site's N."""
    site_ids = np.asarray(site_ids)
    if site_ids.ndim != 1 or not len(site_ids) or replicates < 2:
        raise ValueError("Require subjects and at least two bootstrap replicates")
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(site_ids == site) for site in np.unique(site_ids)]
    return np.concatenate([rng.choice(g, (replicates, len(g)), replace=True) for g in groups], axis=1)


def interval(estimate, draws):
    values = np.asarray(draws, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return {
        "estimate": float(estimate) if np.isfinite(estimate) else None,
        "ci95": [float(v) for v in np.quantile(finite, [.025, .975])] if len(finite) == len(values) and len(values) else None,
        "bootstrap_valid": int(len(finite)),
        "bootstrap_total": int(len(values)),
    }


def summarize_subjects(values, indices):
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    # Never hide exact identities or undefined values by reporting a finite-subset mean.
    if finite.all():
        result = interval(values.mean(), values[indices].mean(axis=1))
    else:
        result = interval(float("nan"), np.full(len(indices), np.nan))
    result.update(n=int(len(values)), n_finite=int(finite.sum()),
                  n_posinf=int(np.isposinf(values).sum()), n_undefined=int(np.isnan(values).sum()),
                  finite_only_mean=float(values[finite].mean()) if finite.any() else None,
                  status="complete" if finite.all() else "nonfinite_values_present")
    return result


def balanced_accuracy_draws(labels, predictions, indices):
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    correct = (labels == predictions).astype(float)
    classes = np.unique(labels)
    estimate = np.mean([correct[labels == site].mean() for site in classes])
    sampled_labels = labels[indices]
    sampled_correct = correct[indices]
    recalls = [np.sum(sampled_correct * (sampled_labels == site), axis=1) /
               np.sum(sampled_labels == site, axis=1) for site in classes]
    return float(estimate), np.mean(recalls, axis=0)
