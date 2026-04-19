# METRICS.md — HarmonIt Evaluation Metrics

This document defines the evaluation metrics used in HarmonIt. The goal is to (i) quantify structural preservation, (ii) measure semantic consistency, and (iii) assess distribution alignment across sites after harmonization.

Unlike the site-probe (model-based evaluation), these metrics operate directly on data (images or features) and do not require a trained classifier.

## Evaluation philosophy

We separate evaluation into two complementary levels:

Model-based evaluation (site-probe): measures how much site information remains via classification accuracy.
Data-based evaluation (this document): measures intrinsic similarity between images, features, and distributions.

This document focuses on the second level.

## Metric groups
*1. Structural metrics*
### Peak Signal-to-Noise Ratio (PSNR)

Measures pixel-wise similarity between two images.

PSNR is defined as: PSNR = 10 log10 (MAX² / MSE)

where MSE is the mean squared error between images.

Detects low-level intensity differences.
Useful for verifying that harmonization does not introduce excessive distortion.

Limitations:

Sensitive to intensity scaling.
Does not capture perceptual or anatomical similarity.

Interpretation:

High PSNR → images are very similar
Low PSNR → strong differences

*2. Feature-level metrics*

These metrics operate on feature representations rather than raw pixels.

### Feature similarity (cosine similarity)

Measures similarity between feature vectors.

Captures higher-level structure beyond pixel intensity.
Approximates semantic similarity when features come from a neural network.

Current implementation:

Uses flattened pixel vectors (proxy features).
Can be replaced with CNN features (e.g., ResNet) for semantic evaluation.

Interpretation:

1 → identical
0 → orthogonal (unrelated)
<0 → opposite

### Cross-correlation
Measures linear correlation between two signals.

Captures structural alignment between images.
Less sensitive to global intensity scaling than PSNR.

Interpretation:

1 → perfectly correlated
0 → no linear relationship
-1 → inverse correlation

*3. Distribution metrics*

These metrics compare statistical distributions of intensities or features.

### Kullback–Leibler (KL) divergence

Measures how one distribution diverges from another.

KL(P || Q) = Σ P(x) log(P(x) / Q(x))

Quantifies distribution mismatch across sites.
Useful for evaluating harmonization effectiveness.

Limitations:

Not symmetric.
Sensitive to zero probabilities.

Interpretation:

0 → identical distributions
Higher → more divergence

Reference:

### Wasserstein distance (Earth Mover’s Distance)

Measures the minimal “cost” to transform one distribution into another.

More stable than KL divergence.
Handles non-overlapping distributions better.

Interpretation:

0 → identical distributions
Larger → more different