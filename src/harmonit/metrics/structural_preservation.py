"""Pixel preservation on a fixed intensity scale."""
import numpy as np


def compute_psnr(img1, img2, data_range=1.0):
    """PSNR for normalized MRI; never infer scale from model output."""
    a, b = np.asarray(img1, dtype=float), np.asarray(img2, dtype=float)
    if a.shape != b.shape or not a.size:
        raise ValueError("Images must be nonempty and have the same shape")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Images must be finite")
    if not np.isfinite(data_range) or data_range <= 0:
        raise ValueError("data_range must be finite and positive")
    mse = np.mean((a - b) ** 2)
    return float('inf') if mse == 0 else float(10 * np.log10(data_range ** 2 / mse))
