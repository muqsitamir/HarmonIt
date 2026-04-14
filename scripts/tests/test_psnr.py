import numpy as np
from harmonit.metrics.structural_preservation import compute_psnr

# Case 1: identical images
img1 = np.random.rand(64, 64)
img2 = img1.copy()

psnr_same = compute_psnr(img1, img2)
print("PSNR (identical):", psnr_same)

# Case 2: noisy image
noise = np.random.normal(0, 0.1, img1.shape)
img_noisy = img1 + noise

psnr_noisy = compute_psnr(img1, img_noisy)
print("PSNR (noisy):", psnr_noisy)

# Case 3: very different image
img_random = np.random.rand(64, 64)
psnr_diff = compute_psnr(img1, img_random)
print("PSNR (random):", psnr_diff)