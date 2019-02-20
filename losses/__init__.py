import numpy as np


def cal_psnr(img1, img2):
    mse = (np.abs(img1 - img2) ** 2).mean()
    psnr = 10 * np.log10(255 * 255 / mse)
    return psnr
