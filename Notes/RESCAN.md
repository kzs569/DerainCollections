RESCAN
======


图像中雨水条纹会严重降低能见度，导致许多当前的计算机视觉算法无法工作，比如在自动驾驶场景下图像去雨就变得非常重要。本文提出了一种基于深度卷积和递归神经网络的新型深度网络体系结构，用于单图像去雨。



该文对图像去雨的多个挑战分别提出了解决方案，取得了非常显著的算法改进。

1.由于背景信息对于定位雨水位置非常重要，该文首先使用了扩张卷积神经网络（dilated convolutional neural network ）来获取大的感受野，同时为更好地适应去雨任务修改了扩张卷积网络。

2.在大雨的图像中，雨水条纹有各种方向和形状，此时将其看作是多个雨水层的叠加。通过结合squeeze-and-excitation模块，根据强度和透明度为不同的雨水条纹层分配不同的α值。

3.由于雨水条纹层彼此重叠，因此在一个stage中不容易完全除去雨水。因此进一步将雨水分解分为多个stage。结合递归神经网络保留先前阶段中的有用信息并有利于后期的除雨。

最终算法在合成数据集和真实数据集上进行了大量实验，结果显示该文提出的方法在所有评估指标下都优于目前的state-of-the-art方法。

SCAN

结合扩展卷积和squeeze-and-excitation模块的单stage的网络结构

![](http://static.extremevision.com.cn/donkey_8c51bd29-6a1f-4937-abea-8bbc7bc92103.jpg)

SCAN的细节参数

![](http://static.extremevision.com.cn/donkey_4c7571f0-b5e7-4ff1-a583-2d515397ebd8.jpg)

结合RNN的多个stage的去雨架构。

![](http://static.extremevision.com.cn/donkey_f210561a-3acf-4ea8-890e-9f8aacd745ff.jpg)


作者实现的SE Block
```python
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
```

其中一个dilation networrk based on gru
```python
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
```
其输出的值和隐状态均是经过一层卷积之后的结果，dilation