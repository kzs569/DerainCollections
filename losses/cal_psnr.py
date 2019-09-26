import torch
import torch.nn as nn


class PSNR(nn.Module):
    def __init__(self):
        super(PSNR, self).__init__()

    def forward(self, img1, img2):
        mse = (torch.abs_(img1 - img2) ** 2).mean()
        psnr = 10 * torch.log10_(255 * 255 / mse)
        return psnr