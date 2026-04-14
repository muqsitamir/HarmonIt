import numpy as np
from harmonit.metrics.structural_preservation import compute_ssim

# Case 1: identical images
img1 = np.random.rand(64, 64)
img2 = img1.copy()

ssim_same = compute_ssim(img1, img2)
print("SSIM (identical):", ssim_same)

# Case 2: noisy version
noise = np.random.normal(0, 0.1, img1.shape)
img_noisy = img1 + noise

ssim_noisy = compute_ssim(img1, img_noisy)
print("SSIM (noisy):", ssim_noisy)

img_random = np.random.rand(64, 64)
print("SSIM (random):", compute_ssim(img1, img_random))