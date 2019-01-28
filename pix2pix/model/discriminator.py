import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, input_nc, output_nc, ndf):
        super(Discriminator, self).__init__()
        # 256 x 256
        self.layer1 = nn.Sequential(nn.Conv2d(input_nc + output_nc, ndf, kernel_size=4, stride=2, padding=1),
                                    nn.LeakyReLU(0.2, inplace=False))
        # 128 x 128
        self.layer2 = nn.Sequential(nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
                                    nn.BatchNorm2d(ndf * 2),
                                    nn.LeakyReLU(0.2, inplace=False))
        # 64 x 64
        self.layer3 = nn.Sequential(nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
                                    nn.BatchNorm2d(ndf * 4),
                                    nn.LeakyReLU(0.2, inplace=False))
        # 32 x 32
        self.layer4 = nn.Sequential(nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1),
                                    nn.BatchNorm2d(ndf * 8),
                                    nn.LeakyReLU(0.2, inplace=False))
        # 31 x 31
        self.layer5 = nn.Sequential(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1),
                                    nn.Sigmoid())
        # 30 x 30

    def forward(self, x):
        out_1 = self.layer1(x)
        out_2 = self.layer2(out_1)
        out_3 = self.layer3(out_2)
        out_4 = self.layer4(out_3)
        out_5 = self.layer5(out_4)
        return out_5
