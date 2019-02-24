Perceptual Adversarial Network 感知对抗网络
========================================

## 1 Introduction
首先介绍Img2Img的任务，主要包括超分辨率，语义分割，图像上色等。 那么对于Pixel-wisely的图像任务，L1和L2 norm来计算output和groundtruth尽管能产生合理的图片，但同时也会产生不可忽视的缺陷，如丢失高频信息造成的模糊，丢失感知信息造成的artifacts

而GAN，cGAN可以较好地生成更真实的图片，也有将像素级的loss和GAN loss结合的方法。然后介绍了 perceptual loss，这种loss可以通过penalizing the discrepancy between extracted high-level features, these models are trained to transform the input image into the output which has same high-level features with the corresponding ground-truth. 也就是可以生成具有相同的高阶特征的图片，显然这个可以做风格迁移，也可以做artifact的压制。

上述的所有loss都从不同方面惩罚了输出和真实图像之间的discrepancy，然而单一的loss还不够，所以需要多个loss结合起来。而perceptual loss的优点是，可以再各个方面进行优化，自动持续的寻找还没有被优化的discrepancy。 the perceptual adversarial loss provides a strategy to penalize the discrepancy between the output and ground-truth images from as many perspectives as possible。

提出了principled perceptual adversarial loss ，利用判别器的隐层来评价output和groundtruth。另外，把pan loss 和gan loss结合，并且在各种image-to-image的任务下做了评估。

## 2 Related Work

1. Image-to-image transformation with feed-forward CNNs

    Super Resolution, Image de-raining\de-snowing, Image inpainting, Semantic Segmentation, Image synthesize, Image colorization, Depth estimations
2.  GANs-based works

    InfoGAN, WGAN, Energy-based GAN, PGN, SRGAN, ID-CGAN, iGAN, IAN, Context-Encoder, pix2pix-cGANs
3. Perceptual loss

    the high-level features extracted from a well-trained image classification network have the capability to capture some perceptual information from the real-world images.
    
    the hidden representations directly extracted by the well-trained image classifier can be regarded as semantic content of the input image.
    
## 3 Methods

![](https://img-blog.csdn.net/2018032716512413?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2Vkb2dhd2FjaGlh/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

作为GAN网络，GAN loss和传统的GAN一样：

<img src="https://latex.codecogs.com/gif.latex?minmaxE_{x\in&space;\chi_{data}}[logD(x)]&space;&plus;&space;E_{x\in&space;\mathbb{Z}}[log(1-D(G(z)))]" title="minmaxE_{x\in \chi_{data}}[logD(x)] + E_{x\in \mathbb{Z}}[log(1-D(G(z)))]" />

不同的传统的GAN的输入是随机初始化的噪声，而这里的输入是待derain, desnow 等的图片，ground-truth则是去雨后的图片。

对Img2Img转换任务来说，输出图片和ground-truth图片在任何高维空间都是都应该保持一致的。

给一对数据
<img src="https://latex.codecogs.com/gif.latex?(x,y)\in(\chi_{input},&space;y_{ground-truth})" title="(x,y)\in(\chi_{input}, y_{ground-truth})" />
，和正margin m，perceptual adversarial loss可以写为：

<img src="https://latex.codecogs.com/gif.latex?L_T(x,y)=\sum^N_{i=1}\lambda_iP_i(T(x),y)" title="L_T(x,y)=\sum^N_{i=1}\lambda_iP_i(T(x),y)" />

<img src="https://latex.codecogs.com/gif.latex?L_P(x,y)=[m-\sum^N_{i=1}\lambda_iP_i(T(x),y)]^&plus;" title="L_P(x,y)=[m-\sum^N_{i=1}\lambda_iP_i(T(x),y)]^+" />

其中
<img src="https://latex.codecogs.com/gif.latex?[\cdot&space;]^&plus;=max(0,\cdot)" title="[\cdot ]^+=max(0,\cdot)" />
，
Pi()衡量了辨别器第i隐层的提取出来的高阶特征的差异

论文中使用L1距离来衡量两者差距：

<img src="https://latex.codecogs.com/gif.latex?P_i(T(x),y)=||H_i(y)-H_i(T(x))||" title="P_i(T(x),y)=||H_i(y)-H_i(T(x))||" />

这样最终的loss函数就是

<img src="https://latex.codecogs.com/gif.latex?J_T=log(1-D(T(x)))&plus;\sum_i\lambda_iP_i(T(x),y)" title="J_T=log(1-D(T(x)))+\sum_i\lambda_iP_i(T(x),y)" />

<img src="https://latex.codecogs.com/gif.latex?J_D=-log(D(y))-log(1-D(T(x)))&plus;[m-\sum_i\lambda_iP_i(T(x),y)]^&plus;" title="J_D=-log(D(y))-log(1-D(T(x)))+[m-\sum_i\lambda_iP_i(T(x),y)]^+" />

G和D的loss 函数分别代表的意义如下：

T希望我们生成的图像能够在D的判决下更趋向于1，也就是True，而且希望T(x)和y的perceptual的距离更加接近。

而D刚好相反，一方面希望两者尽量分开，另一方面希望两者perceptual距离更大。但是m是一个margin，超过了这一项就变成0，并且没有gradient了。

- Network Architectures

基本可以看出Generator是一个U-Net的结构

![](https://img-blog.csdn.net/20180327165154576?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2Vkb2dhd2FjaGlh/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

Discriminator是一个图像分类的CNN结构

![](https://img-blog.csdn.net/20180327165233996?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2Vkb2dhd2FjaGlh/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

综上，可以发现Perceputal Adversarial Net与Pix2Pix的区别就在于Discriminator，pix2pix的Discriminator就是一个CNN+FC的分类结构，PAN为了更多考虑低阶特征，对loss的计算进行了优化。

- Inference
[【文献阅读】Perceptual Generative Adversarial Networks for Small Object Detection –CVPR-2017](https://blog.csdn.net/u011995719/article/details/76615649)