import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import structural_similarity as ssim_fn

def psnr(sr: np.ndarray, hr: np.ndarray, max_val: float = 1.0) -> float:
    return compare_psnr(hr, sr, data_range=max_val)


def ssim(sr: np.ndarray, hr: np.ndarray, data_range: float = 1.0) -> float:
    return ssim_fn(hr, sr, data_range=data_range, channel_axis=2)