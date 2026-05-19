import numpy as np

try:
    from skimage.metrics import peak_signal_noise_ratio as psnr
except ImportError:  # pragma: no cover - exercised on lean remote envs
    psnr = None

def compute_psnr(img1, img2):
    data_range = float(np.max(img2) - np.min(img2))
    if psnr is not None:
        return psnr(img1, img2, data_range=data_range)
    mse = float(np.mean((np.asarray(img1) - np.asarray(img2)) ** 2))
    if mse <= 0.0:
        return float("inf")
    return 20.0 * np.log10(max(data_range, 1e-8)) - 10.0 * np.log10(mse)
