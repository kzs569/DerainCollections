import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

import settings.rescan as settings


class SEBlock(nn.Module):
    def __init__(self, input_dim, reduction):
        super().__init__()
        mid = int(input_dim / reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, reduction),
            nn.ReLU(inplace=True),
            nn.Linear(reduction, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class NoSEBlock(nn.Module):
    def __init__(self, input_dim, reduction):
        super().__init__()

    def forward(self, x):
        return x


SE = SEBlock if settings.use_se else NoSEBlock


class ConvDirec(nn.Module):
    def __init__(self, inp_dim, oup_dim, kernel, dilation):
        super().__init__()
        pad = int(dilation * (kernel - 1) / 2)
        self.conv = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad, dilation=dilation)
        self.se = SE(oup_dim, 6)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x, h=None):
        x = self.conv(x)
        x = self.relu(self.se(x))
        return x, None


class ConvRNN(nn.Module):
    def __init__(self, inp_dim, oup_dim, kernel, dilation):
        super().__init__()
        pad_x = int(dilation * (kernel - 1) / 2)
        self.conv_x = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad_x, dilation=dilation)

        pad_h = int((kernel - 1) / 2)
        self.conv_h = nn.Conv2d(oup_dim, oup_dim, kernel, padding=pad_h)

        self.se = SE(oup_dim, 6)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x, h=None):
        if h is None:
            h = F.tanh(self.conv_x(x))
        else:
            h = F.tanh(self.conv_x(x) + self.conv_h(h))

        h = self.relu(self.se(h))
        return h, h


class ConvGRU(nn.Module):
    def __init__(self, inp_dim, oup_dim, kernel, dilation):
        super().__init__()
        pad_x = int(dilation * (kernel - 1) / 2)
        self.conv_xz = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad_x, dilation=dilation)
        self.conv_xr = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad_x, dilation=dilation)
        self.conv_xn = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad_x, dilation=dilation)

        pad_h = int((kernel - 1) / 2)
        self.conv_hz = nn.Conv2d(oup_dim, oup_dim, kernel, padding=pad_h)
        self.conv_hr = nn.Conv2d(oup_dim, oup_dim, kernel, padding=pad_h)
        self.conv_hn = nn.Conv2d(oup_dim, oup_dim, kernel, padding=pad_h)

        self.se = SE(oup_dim, 6)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x, h=None):
        if h is None:
            z = F.sigmoid(self.conv_xz(x))
            f = F.tanh(self.conv_xn(x))
            h = z * f
        else:
            z = F.sigmoid(self.conv_xz(x) + self.conv_hz(h))
            r = F.sigmoid(self.conv_xr(x) + self.conv_hr(h))
            n = F.tanh(self.conv_xn(x) + self.conv_hn(r * h))
            h = (1 - z) * h + z * n

        h = self.relu(self.se(h))
        return h, h


class ConvLSTM(nn.Module):
    def __init__(self, inp_dim, oup_dim, kernel, dilation):
        super().__init__()
        pad_x = int(dilation * (kernel - 1) / 2)
        self.conv_xf = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad_x, dilation=dilation)
        self.conv_xi = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad_x, dilation=dilation)
        self.conv_xo = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad_x, dilation=dilation)
        self.conv_xj = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad_x, dilation=dilation)

        pad_h = int((kernel - 1) / 2)
        self.conv_hf = nn.Conv2d(oup_dim, oup_dim, kernel, padding=pad_h)
        self.conv_hi = nn.Conv2d(oup_dim, oup_dim, kernel, padding=pad_h)
        self.conv_ho = nn.Conv2d(oup_dim, oup_dim, kernel, padding=pad_h)
        self.conv_hj = nn.Conv2d(oup_dim, oup_dim, kernel, padding=pad_h)

        self.se = SE(oup_dim, 6)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x, pair=None):
        if pair is None:
            i = F.sigmoid(self.conv_xi(x))
            o = F.sigmoid(self.conv_xo(x))
            j = F.tanh(self.conv_xj(x))
            c = i * j
            h = o * c
        else:
            h, c = pair
            f = F.sigmoid(self.conv_xf(x) + self.conv_hf(h))
            i = F.sigmoid(self.conv_xi(x) + self.conv_hi(h))
            o = F.sigmoid(self.conv_xo(x) + self.conv_ho(h))
            j = F.tanh(self.conv_xj(x) + self.conv_hj(h))
            c = f * c + i * j
            h = o * F.tanh(c)

        h = self.relu(self.se(h))
        return h, [h, c]


RecUnit = {
    'Conv': ConvDirec,
    'RNN': ConvRNN,
    'GRU': ConvGRU,
    'LSTM': ConvLSTM,
}[settings.uint]


class RESCAN(nn.Module):
    def __init__(self):
        super().__init__()
        channel = settings.channel

        self.rnns = nn.ModuleList(
            [RecUnit(3, channel, 3, 1)] +
            [RecUnit(channel, channel, 3, 2 ** i) for i in range(settings.depth - 3)]
        )

        self.dec = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1),
            SE(channel, 6),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channel, 3, 1),
        )

    def forward(self, x):
        ori = x
        old_states = [None for _ in range(len(self.rnns))]
        oups = []

        for i in range(settings.stage_num):
            states = []
            for rnn, state in zip(self.rnns, old_states):
                x, st = rnn(x, state)
                states.append(st)
            x = self.dec(x)

            if settings.frame == 'Add' and i > 0:
                x = x + Variable(oups[-1].data)

            oups.append(x)
            old_states = states.copy()
            x = ori - x

        return oups


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf):
        super(Generator, self).__init__()
        # 256 x 256
        self.e1 = nn.Sequential(nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1),
                                nn.LeakyReLU(0.2, inplace=True))
        # 128 x 128
        self.e2 = nn.Sequential(nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf * 2),
                                nn.LeakyReLU(0.2, inplace=True))
        # 64 x 64
        self.e3 = nn.Sequential(nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf * 4),
                                nn.LeakyReLU(0.2, inplace=True))
        # 32 x 32
        self.e4 = nn.Sequential(nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf * 8),
                                nn.LeakyReLU(0.2, inplace=True))
        # 16 x 16
        self.e5 = nn.Sequential(nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf * 8),
                                nn.LeakyReLU(0.2, inplace=True))
        # 8 x 8
        self.e6 = nn.Sequential(nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf * 8),
                                nn.LeakyReLU(0.2, inplace=True))
        # 4 x 4
        self.e7 = nn.Sequential(nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf * 8),
                                nn.LeakyReLU(0.2, inplace=True))
        # 2 x 2
        self.e8 = nn.Sequential(nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
                                nn.LeakyReLU(0.2, inplace=True))
        # 1 x 1
        self.d1 = nn.Sequential(nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf * 8),
                                nn.Dropout())
        # 2 x 2
        self.d2 = nn.Sequential(nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf * 8),
                                nn.Dropout())
        # 4 x 4
        self.d3 = nn.Sequential(nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf * 8),
                                nn.Dropout())
        # 8 x 8
        self.d4 = nn.Sequential(nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf * 8))
        # 16 x 16
        self.d5 = nn.Sequential(nn.ConvTranspose2d(ngf * 8 * 2, ngf * 4, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf * 4))
        # 32 x 32
        self.d6 = nn.Sequential(nn.ConvTranspose2d(ngf * 4 * 2, ngf * 2, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf * 2))
        # 64 x 64
        self.d7 = nn.Sequential(nn.ConvTranspose2d(ngf * 2 * 2, ngf, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(ngf))
        # 128 x 128
        self.d8 = nn.ConvTranspose2d(ngf * 2, output_nc, kernel_size=4, stride=2, padding=1)
        # 256 x 256
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # encoder
        out_e1 = self.e1(x)  # 128 x 128
        out_e2 = self.e2(out_e1)  # 64 x 64
        out_e3 = self.e3(out_e2)  # 32 x 32
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
        out_d6 = self.d6(self.relu(out_d5_))  # 64 x 64
        out_d6_ = torch.cat((out_d6, out_e2), 1)
        out_d7 = self.d7(self.relu(out_d6_))  # 128 x 128
        out_d7_ = torch.cat((out_d7, out_e1), 1)
        out_d8 = self.d8(self.relu(out_d7_))  # 256 x 256
        out = self.tanh(out_d8)
        return out


class Discriminator(nn.Module):
    def __init__(self, input_nc, output_nc, ndf):
        super(Discriminator, self).__init__()
        # 256 x 256
        self.layer1 = nn.Sequential(nn.Conv2d(input_nc + output_nc, ndf, kernel_size=4, stride=2, padding=1),
                                    nn.LeakyReLU(0.2, inplace=True))
        # 128 x 128
        self.layer2 = nn.Sequential(nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
                                    nn.BatchNorm2d(ndf * 2),
                                    nn.LeakyReLU(0.2, inplace=True))
        # 64 x 64
        self.layer3 = nn.Sequential(nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
                                    nn.BatchNorm2d(ndf * 4),
                                    nn.LeakyReLU(0.2, inplace=True))
        # 32 x 32
        self.layer4 = nn.Sequential(nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1),
                                    nn.BatchNorm2d(ndf * 8),
                                    nn.LeakyReLU(0.2, inplace=True))
        # 31 x 31
        self.layer5 = nn.Sequential(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1),
                                    nn.Sigmoid())
        # 30 x 30

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return out


if __name__ == '__main__':
    ts = torch.Tensor(16, 3, 64, 64)
    vr = Variable(ts)
    net = RESCAN()
    print(net)
    print(net.rnns, len(net.rnns))
    oups = net(vr)
    for oup in oups:
        print(oup.size())
