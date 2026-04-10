# SSIM
from skimage.metrics import structural_similarity as ssim

def compute_ssim(img1, img2):
    return ssim(img1, img2, data_range=img2.max() - img2.min())

# PSNR
from skimage.metrics import peak_signal_noise_ratio as psnr

def compute_psnr(img1, img2):
    return psnr(img1, img2, data_range=img2.max() - img2.min())