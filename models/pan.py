import torch.nn as nn
import torch
from torch.autograd import Variable

class PAN_Discriminator(nn.Module):
    def __init__(self, input_nc, output_nc, ndf):
        super(PAN_Discriminator, self).__init__()
        self.intermediate_outputs = []
        # 64 x 64
        self.layer1 = nn.Sequential(nn.Conv2d(input_nc + output_nc, ndf, kernel_size=4, stride=2, padding=1),
                                    nn.BatchNorm2d(ndf),
                                    nn.LeakyReLU(0.2, inplace=False))
        self.layer1.register_forward_hook(self.add_intermediate_output)
        # 32 x 32
        self.layer2 = nn.Sequential(nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=1, padding=1),
                                    nn.BatchNorm2d(ndf * 2),
                                    nn.LeakyReLU(0.2, inplace=False))
        self.layer2.register_forward_hook(self.add_intermediate_output)
        # 31 x 31
        self.layer3 = nn.Sequential(nn.Conv2d(ndf * 2, 1, kernel_size=4, stride=1, padding=1),
                                    nn.Sigmoid())
        self.layer3.register_forward_hook(self.add_intermediate_output)
        # 30 x 30

    def forward(self, x):
        self.intermediate_outputs = []
        out_3 = self.layer1(x)
        out_4 = self.layer2(out_3)
        out_5 = self.layer3(out_4)
        return out_5

    def add_intermediate_output(self, conv, input, output):
        self.intermediate_outputs.append(Variable(output.data, requires_grad=False))

    def get_intermediate_outputs(self):
        return self.intermediate_outputs
