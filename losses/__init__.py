import numpy as np
import torch.nn as nn


def cal_psnr(img1, img2):
    mse = (np.abs(img1 - img2) ** 2).mean()
    psnr = 10 * np.log10(255 * 255 / mse)
    return psnr


def get_critical(name):
    """get_critical

    :param name:
    """

    if name is None or name == '':
        return nn.MSELoss

    return {
        "l1loss": nn.L1Loss,
        "mseloss": nn.MSELoss,
    }[name]
