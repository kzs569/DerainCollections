博客要点记录
================
- 2018.10.18

*https://mp.weixin.qq.com/s/YPdIfrCQrWGYqtREIl3OaA*

17年超分大赛NTIRE 的冠军 EDSR 也是基于 SRGAN 的变体

SRGAN 是基于 GAN 方法进行训练的，有一个生成器和一个判别器，判别器的主体使用 VGG19，生成器是一连串的 Residual block 连接，同时在模型后部也加入了 subpixel 模块，借鉴了 Shi et al 的 Subpixel Network [6] 的思想，让图片在最后面的网络层才增加分辨率，提升分辨率的同时减少计算资源消耗。

WGAN解决了没有loss指标来判别Generator和Discriminator停止训练的问题。WGAN 使用 Wasserstein 距离来描述两个数据集分布之间的差异程度，只要把模型修改成 WGAN 的形式，就能根据一个唯一的 loss 来监控模型训练的程度。

    · 判别器最后一层去掉 sigmoid

    · 生成器和判别器的 loss 不取 log

    · 每次更新判别器的参数之后把它们的绝对值截断到不超过一个固定常数 c

    · 不要用基于动量的优化算法（包括 momentum 和 Adam），推荐 RMSProp，SGD 也行

    --来自《令人拍案叫绝的Wasserstein GAN》

SRGAN With Wasserstein GAN的改造

1.对模型进行了WGAN的改造 2.增加了Tensorboard，监控loss下降情况 3.对作者model.py中，Generator的最后一层卷积层的kernal从1*1改为9*9

提到的工业界问题

在实际生产使用中，遇到的低分辨率图片并不一定都是 PNG 格式的（无损压缩的图片复原效果最好），而且会带有不同程度的失真（有损压缩导致的 artifacts）。笔者尝试过很多算法，例如 SRGAN、EDSR、RAISR、Fast Neural Style 等等，这类图片目前使用任何一种超分算法都没法在提高分辨率的同时消除失真。



这个问题我在 @董豪 SRGAN 项目的 issue 中也讨论过，同时在知乎也提出过这个问题：SRGAN 超分辨率方法对于低清 jpg 格式的图片复原效果是否比不上对低清 png 格式的复原效果？



可惜没有很好的答案。目前学术界貌似还没有很好的算法，这里欢迎各位在评论区或者 Github 上来讨论。


提到一篇SubPixel的文章 [6]. Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network

https://www.paperweekly.site/papers/386

[2]. Is the deconvolution layer the same as a convolutional layer?

https://arxiv.org/abs/1609.07009


- 2019.01.28

- CycleGAN 

[CycleGAN论文笔记](https://www.paperweekly.site/papers/notes/233)

