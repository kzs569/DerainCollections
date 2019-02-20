import torch.nn as nn
import torch
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable
from models.utils import BottleneckBlock, TransitionBlock


class vgg19ca(nn.Module):
    def __init__(self):
        super(vgg19ca, self).__init__()
        ############# 256-256  ##############
        vgg19 = models.vgg19_bn(pretrained=True)

        self.feature1 = nn.Sequential(vgg19.features[0])
        for i in range(1, 3):
            self.feature1.add_module(str(i), vgg19.features[i])

        self.feature2 = nn.Sequential(nn.Conv2d(64, 24, kernel_size=3, stride=1, padding=1),  # 1mm
                                      nn.ReLU(inplace=True),
                                      nn.AvgPool2d(kernel_size=7))
        self.fc1 = nn.Linear(127896, 512)  # 这个数据一定要改
        self.fc2 = nn.Linear(512, 4)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.feature1(x)
        out = self.feature2(out)

        out = out.view(out.size(0), -1)

        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out


class DenseDerain(nn.Module):
    def __init__(self, cfg):
        super(DenseDerain, self).__init__()

        self.dense1 = DenseBase3()
        self.dense2 = DenseBase5()
        self.dense3 = DenseBase7()

        self.refine1 = nn.Conv2d(47, 47, 3, 1, 1)

        self.tanh = nn.Tanh()

        self.conv1010 = nn.Conv2d(47, 2, kernel_size=1, stride=1, padding=0)  # 1mm
        self.conv1020 = nn.Conv2d(47, 2, kernel_size=1, stride=1, padding=0)  # 1mm
        self.conv1030 = nn.Conv2d(47, 2, kernel_size=1, stride=1, padding=0)  # 1mm
        self.conv1040 = nn.Conv2d(47, 2, kernel_size=1, stride=1, padding=0)  # 1mm

        self.refine2 = nn.Conv2d(47 + 8, 3, kernel_size=3, stride=1, padding=1)

        self.refineclean1 = nn.Conv2d(3, 8, kernel_size=7, stride=1, padding=3)
        self.refineclean2 = nn.Conv2d(8, 3, kernel_size=3, stride=1, padding=1)

        self.upsample = F.upsample_nearest
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.cfg = cfg

    def forward(self, x, label_d):
        x3 = self.dense3(x)
        x2 = self.dense2(x)
        x1 = self.dense1(x)

        label = label_d.unsqueeze_(1).unsqueeze_(2).unsqueeze_(3)
        label = label.repeat(1, 8, self.cfg['data']['patch_size'], self.cfg['data']['patch_size'])
        label = Variable(label.float(), requires_grad=False)

        x4 = torch.cat([x, x1, x2, x3, label], 1)

        x5 = self.relu(self.refine1(x4))

        shape_out = x5.data.size()
        shape_out = shape_out[2:4]

        x101 = F.avg_pool2d(x5, 32)
        x102 = F.avg_pool2d(x5, 16)
        x103 = F.avg_pool2d(x5, 8)
        x104 = F.avg_pool2d(x5, 4)

        x1010 = self.upsample(self.relu(self.conv1010(x101)), size=shape_out)
        x1020 = self.upsample(self.relu(self.conv1020(x102)), size=shape_out)
        x1030 = self.upsample(self.relu(self.conv1030(x103)), size=shape_out)
        x1040 = self.upsample(self.relu(self.conv1040(x104)), size=shape_out)

        dehaze = torch.cat((x1010, x1020, x1030, x1040, x5), 1)
        residual = self.tanh(self.refine2(dehaze))
        background = x - residual
        clean = self.relu(self.refineclean1(background))
        clean = self.tanh(self.refineclean2(clean))

        return residual, clean


class DenseBase3(nn.Module):
    def __init__(self):
        super(DenseBase3, self).__init__()
        # 256x256
        self.dense_trans_block1 = nn.Sequential(BottleneckBlock(in_planes=3, out_planes=5, kernelSize=3),
                                                TransitionBlock(in_planes=8, out_planes=4),
                                                nn.AvgPool2d(kernel_size=2))
        # 128x128
        self.dense_trans_block2 = nn.Sequential(BottleneckBlock(in_planes=4, out_planes=8, kernelSize=3),
                                                TransitionBlock(in_planes=12, out_planes=12))
        # 128x128
        self.dense_trans_block3 = nn.Sequential(BottleneckBlock(in_planes=12, out_planes=4, kernelSize=3),
                                                TransitionBlock(in_planes=16, out_planes=12))
        # 128x128
        self.dense_trans_block4 = nn.Sequential(BottleneckBlock(in_planes=12, out_planes=4, kernelSize=3),
                                                TransitionBlock(in_planes=16, out_planes=12))
        # 128x128
        self.dense_trans_block5 = nn.Sequential(BottleneckBlock(in_planes=24, out_planes=8, kernelSize=3),
                                                TransitionBlock(in_planes=32, out_planes=4))
        # 128x128
        self.dense_trans_block6 = nn.Sequential(BottleneckBlock(in_planes=8, out_planes=8, kernelSize=3),
                                                TransitionBlock(in_planes=16, out_planes=4),
                                                nn.Upsample(scale_factor=2, mode='nearest'))
        # 256x256
        self.conv1 = nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1)  # 1mm
        self.conv2 = nn.Conv2d(12, 1, kernel_size=3, stride=1, padding=1)  # 1mm
        self.conv3 = nn.Conv2d(12, 1, kernel_size=3, stride=1, padding=1)  # 1mm
        self.conv4 = nn.Conv2d(24, 1, kernel_size=3, stride=1, padding=1)  # 1mm
        self.conv5 = nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1)  # 1mm
        self.conv6 = nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1)  # 1mm
        self.upsample = F.upsample_nearest
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.dense_trans_block1(x)
        x2 = self.dense_trans_block2(x1)
        x3 = self.dense_trans_block3(x2)
        x4 = self.dense_trans_block4(x3)
        # x4 = x4 + x2
        x4 = torch.cat([x4, x2], 1)
        x5 = self.dense_trans_block5(x4)
        # x5 = x5 + x1
        x5 = torch.cat([x5, x1], 1)
        x6 = self.dense_trans_block6(x5)

        shape_out = x6.data.size()
        shape_out = shape_out[2:4]
        x11 = self.upsample(self.relu(self.conv1(x1)), size=shape_out)
        x21 = self.upsample(self.relu(self.conv2(x2)), size=shape_out)
        x31 = self.upsample(self.relu(self.conv3(x3)), size=shape_out)
        x41 = self.upsample(self.relu(self.conv4(x4)), size=shape_out)
        x51 = self.upsample(self.relu(self.conv5(x5)), size=shape_out)

        x6 = torch.cat([x6, x51, x41, x31, x21, x11, x], 1)

        return x6


class DenseBase5(nn.Module):
    def __init__(self):
        super(DenseBase5, self).__init__()
        # 256x256
        self.dense_trans_block1 = nn.Sequential(BottleneckBlock(in_planes=3, out_planes=13, kernelSize=5),
                                                TransitionBlock(in_planes=16, out_planes=8),
                                                nn.AvgPool2d(kernel_size=2))
        # 128x128
        self.dense_trans_block2 = nn.Sequential(BottleneckBlock(in_planes=8, out_planes=16, kernelSize=5),
                                                TransitionBlock(in_planes=24, out_planes=16),
                                                nn.AvgPool2d(kernel_size=2))
        # 64x64
        self.dense_trans_block3 = nn.Sequential(BottleneckBlock(in_planes=16, out_planes=16, kernelSize=5),
                                                TransitionBlock(in_planes=32, out_planes=16))
        # 64x64
        self.dense_trans_block4 = nn.Sequential(BottleneckBlock(in_planes=16, out_planes=16, kernelSize=5),
                                                TransitionBlock(in_planes=32, out_planes=16))
        # 128x128
        self.dense_trans_block5 = nn.Sequential(BottleneckBlock(in_planes=32, out_planes=8, kernelSize=5),
                                                TransitionBlock(in_planes=40, out_planes=8),
                                                nn.Upsample(scale_factor=2, mode='nearest'))
        # 256x256
        self.dense_trans_block6 = nn.Sequential(BottleneckBlock(in_planes=16, out_planes=8, kernelSize=5),
                                                TransitionBlock(in_planes=24, out_planes=4),
                                                nn.Upsample(scale_factor=2, mode='nearest'))
        self.tanh = nn.Tanh()
        self.conv1 = nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1)  # 1mm
        self.conv2 = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)  # 1mm
        self.conv3 = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)  # 1mm
        self.conv4 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)  # 1mm
        self.conv5 = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)  # 1mm

        self.upsample = F.upsample_nearest
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        # 512x512
        x1 = self.dense_trans_block1(x)
        # 256x256
        x2 = self.dense_trans_block2(x1)
        # 128x128
        x3 = self.dense_trans_block3(x2)
        # 128x128
        x4 = self.dense_trans_block4(x3)
        x4 = torch.cat([x4, x2], 1)
        # 256x256
        x5 = self.dense_trans_block5(x4)
        x5 = torch.cat([x5, x1], 1)
        # 512x512
        x6 = self.dense_trans_block6(x5)

        shape_out = x6.data.size()
        shape_out = shape_out[2:4]

        x11 = self.upsample(self.relu((self.conv1(x1))), size=shape_out)
        x21 = self.upsample(self.relu((self.conv2(x2))), size=shape_out)
        x31 = self.upsample(self.relu((self.conv3(x3))), size=shape_out)
        x41 = self.upsample(self.relu((self.conv4(x4))), size=shape_out)
        x51 = self.upsample(self.relu((self.conv5(x5))), size=shape_out)

        x6 = torch.cat([x6, x51, x41, x31, x21, x11, x], 1)

        return x6


class DenseBase7(nn.Module):
    def __init__(self):
        super(DenseBase7, self).__init__()
        # 256x256
        self.dense_trans_block1 = nn.Sequential(BottleneckBlock(in_planes=3, out_planes=13, kernelSize=7),
                                                TransitionBlock(in_planes=16, out_planes=8),
                                                nn.AvgPool2d(kernel_size=2))
        # 128x128
        self.dense_trans_block2 = nn.Sequential(BottleneckBlock(in_planes=8, out_planes=16, kernelSize=7),
                                                TransitionBlock(in_planes=24, out_planes=16),
                                                nn.AvgPool2d(kernel_size=2))
        # 64x64
        self.dense_trans_block3 = nn.Sequential(BottleneckBlock(in_planes=16, out_planes=16, kernelSize=7),
                                                TransitionBlock(in_planes=32, out_planes=16),
                                                nn.AvgPool2d(kernel_size=2))
        # 64x64
        self.dense_trans_block4 = nn.Sequential(BottleneckBlock(in_planes=16, out_planes=16, kernelSize=7),
                                                TransitionBlock(in_planes=32, out_planes=16),
                                                nn.Upsample(scale_factor=2, mode='nearest'))
        # 128x128
        self.dense_trans_block5 = nn.Sequential(BottleneckBlock(in_planes=32, out_planes=8, kernelSize=7),
                                                TransitionBlock(in_planes=40, out_planes=8),
                                                nn.Upsample(scale_factor=2, mode='nearest'))
        # 256x256
        self.dense_trans_block6 = nn.Sequential(BottleneckBlock(in_planes=16, out_planes=8, kernelSize=7),
                                                TransitionBlock(in_planes=24, out_planes=4),
                                                nn.Upsample(scale_factor=2, mode='nearest'))
        self.tanh = nn.Tanh()
        self.conv1 = nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1)  # 1mm
        self.conv2 = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)  # 1mm
        self.conv3 = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)  # 1mm
        self.conv4 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)  # 1mm
        self.conv5 = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)  # 1mm

        self.upsample = F.upsample_nearest
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        # 512x512
        x1 = self.dense_trans_block1(x)
        # 256x256
        x2 = self.dense_trans_block2(x1)
        # 128x128
        x3 = self.dense_trans_block3(x2)
        # 128x128
        x4 = self.dense_trans_block4(x3)
        x4 = torch.cat([x4, x2], 1)
        # 256x256
        x5 = self.dense_trans_block5(x4)
        x5 = torch.cat([x5, x1], 1)
        # 512x512
        x6 = self.dense_trans_block6(x5)

        shape_out = x6.data.size()
        shape_out = shape_out[2:4]

        x11 = self.upsample(self.relu((self.conv1(x1))), size=shape_out)
        x21 = self.upsample(self.relu((self.conv2(x2))), size=shape_out)
        x31 = self.upsample(self.relu((self.conv3(x3))), size=shape_out)
        x41 = self.upsample(self.relu((self.conv4(x4))), size=shape_out)
        x51 = self.upsample(self.relu((self.conv5(x5))), size=shape_out)

        x6 = torch.cat([x6, x51, x41, x31, x21, x11, x], 1)

        return x6


if __name__ == '__main__':
    cfg = {
        'batch_size': 1,
        'data': {
            'patch_size': 512,
        }
    }
    db = DenseDerain(cfg=cfg).cuda()
    print(db)
    x = Variable(torch.Tensor(1, 3, 512, 512).cuda())
    out = db(x, label_d=Variable(torch.FloatTensor(1)))
    print(out.shape)
