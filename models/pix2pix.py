import torch.nn as nn
import torch


class Pix2Pix_Generator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf):
        super(Pix2Pix_Generator, self).__init__()
        # 64 x 64
        self.e3 = nn.Sequential(nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf),
                                nn.LeakyReLU(0.2, inplace=False))
        # 32 x 32
        self.e4 = nn.Sequential(nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf * 2),
                                nn.LeakyReLU(0.2, inplace=False))
        # 16 x 16
        self.e5 = nn.Sequential(nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf * 4),
                                nn.LeakyReLU(0.2, inplace=False))
        # 8 x 8
        self.e6 = nn.Sequential(nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf * 8),
                                nn.LeakyReLU(0.2, inplace=False))
        # 4 x 4
        self.e7 = nn.Sequential(nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf * 8),
                                nn.LeakyReLU(0.2, inplace=False))
        # 2 x 2
        self.e8 = nn.Sequential(nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
                                nn.LeakyReLU(0.2, inplace=False))
        # 1 x 1
        self.d1 = nn.Sequential(nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf * 8),
                                nn.Dropout())
        # 2 x 2
        self.d2 = nn.Sequential(nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf * 8),
                                nn.Dropout())
        # 4 x 4
        self.d3 = nn.Sequential(nn.ConvTranspose2d(ngf * 8 * 2, ngf * 4, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf * 4),
                                nn.Dropout())
        # 8 x 8
        self.d4 = nn.Sequential(nn.ConvTranspose2d(ngf * 4 * 2, ngf * 2, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf * 2))
        # 16 x 16
        self.d5 = nn.Sequential(nn.ConvTranspose2d(ngf * 2 * 2, ngf, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf))
        # 32 x 32
        self.d8 = nn.ConvTranspose2d(ngf * 2, output_nc, kernel_size=4, stride=2, padding=1)
        # # 64 x 64
        self.relu = nn.ReLU(inplace=False)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # encoder
        out_e3 = self.e3(x)
        out_e4 = self.e4(out_e3)  # 16 x 16
        out_e5 = self.e5(out_e4)  # 8 x 8
        out_e6 = self.e6(out_e5)  # 4 x 4
        out_e7 = self.e7(out_e6)  # 2 x 2
        out_e8 = self.e8(out_e7)  # 1 x 1
        # decoder
        out_d1 = self.d1(self.relu(out_e8))  # 2 x 2
        out_d1_ = torch.cat((out_d1, out_e7), 1)
        out_d2 = self.d2(self.relu(out_d1_))  # 4 x 4
        out_d2_ = torch.cat((out_d2, out_e6), 1)
        out_d3 = self.d3(self.relu(out_d2_))  # 8 x 8
        out_d3_ = torch.cat((out_d3, out_e5), 1)
        out_d4 = self.d4(self.relu(out_d3_))  # 16 x 16
        out_d4_ = torch.cat((out_d4, out_e4), 1)
        out_d5 = self.d5(self.relu(out_d4_))  # 32 x 32
        out_d5_ = torch.cat((out_d5, out_e3), 1)
        out_d8 = self.d8(self.relu(out_d5_))  # 256 x 256
        out = self.tanh(out_d8)
        return out


class Pix2Pix_Discriminator(nn.Module):
    def __init__(self, input_nc, output_nc, ndf):
        super(Pix2Pix_Discriminator, self).__init__()
        # 64 x 64
        self.layer3 = nn.Sequential(nn.Conv2d(input_nc + output_nc, ndf, kernel_size=4, stride=2, padding=1),
                                    nn.BatchNorm2d(ndf),
                                    nn.LeakyReLU(0.2, inplace=False))
        # 32 x 32
        self.layer4 = nn.Sequential(nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=1, padding=1),
                                    nn.BatchNorm2d(ndf * 2),
                                    nn.LeakyReLU(0.2, inplace=False))
        # 31 x 31
        self.layer5 = nn.Sequential(nn.Conv2d(ndf * 2, 1, kernel_size=4, stride=1, padding=1),
                                    nn.Sigmoid())
        # 30 x 30

    def forward(self, x):
        out_3 = self.layer3(x)
        out_4 = self.layer4(out_3)
        out_5 = self.layer5(out_4)
        return out_5
