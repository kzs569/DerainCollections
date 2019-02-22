【论文笔记】深度联合单图像雨迹检测和移除 Deep Joint Rain Detection and Removal from a Single Image
===================================

- 简介：多任务全卷积从单张图片中去除雨迹。本文在现有的模型上，开发了一种多任务深度学习框架，学习了三个方面，包括二元雨条纹映射(binary rain streak map)，雨条纹外观和干净的背景。特别是新添加的二元雨条纹映射，其损失函数可以为神经网络提供额外的强特征。对于雨带积累现象（暴雨形成的如烟如雾的现象），采取循环雨检测和清除，以迭代和渐进方式清除。

- 动机：恢复暴雨下拍摄的图像，在自动驾驶等领域是重要的研究问题。在暴雨下拍摄的图像包含背景层和将与条纹层，此任务的目标在于输出干净的背景。

    传统方法主要有两个问题需要解决。1）在一幅图像中雨迹有多个密度。2）一些没有雨部分的去雨会造成过度平滑。

- 雨模型由原来的:

    <img src="https://latex.codecogs.com/gif.latex?O=B&plus;\overline{S}" title="O=B+\overline{S}" />
    
    其中O为捕捉到的图像，B为背景图像，<img src="https://latex.codecogs.com/gif.latex?\overline{S}" title="\overline{S}" />是雨迹图像，变为：

    <img src="https://latex.codecogs.com/gif.latex?O=B&plus;SR" title="O=B+SR" />
    
    其中，B是背景层，S为雨迹层，O为原始图片，R是二元值，1表示雨区，0表示无雨区。这里有两个优点，1）给了网络额外信息去学习下雨的区域2）对于有雨和无雨区域的处理方式是不同的，因此这样可以保存更多图像的细节。
    
- 联合的雨迹检测和去除：

![](http://ww1.sinaimg.cn/large/006ocvumgy1g0f23bksdfj310c0h7drq.jpg)

模型最终采用的方程为<img src="https://latex.codecogs.com/gif.latex?O=\alpha(B&plus;\sum^s_{t=1}\widetilde{S}_tR)&plus;(1-\alpha)A" title="O=\alpha(B+\sum^s_{t=1}\widetilde{S}_tR)+(1-\alpha)A" />，
其中每个<img src="https://latex.codecogs.com/gif.latex?\widetilde{S}_t" title="\widetilde{S}_t" />是同一方向和形状的雨点。s是叠加的数量，A是整体的大气亮度。

雨迹移除的过程如上图所示，首先使用基于上下文扩展的网络（Contextualized Dilated Networks，论文4.2详述）抽取雨迹特征表示F，基于此，接下来依次预测R,S,B。

针对上面的公式，文中建立极大似然法MAP进行处理：

<img src="https://latex.codecogs.com/gif.latex?arg\underset{B,S,R}{min}||O-B-SR||^2_2&plus;P_b(B)&plus;P_s(S)&plus;P_r(R)" title="arg\underset{B,S,R}{min}||O-B-SR||^2_2+P_b(B)+P_s(S)+P_r(R)" />

式子中后三项是三个先验（B、S、R）。采用卷积网络处理，首先通过dilated network来得到雨的特征F，R、S、B分别通过串联。主要流程如下：

1. R is estimated by two convolutions on F
2. S is predicted by a convolution on the concentration <img src="https://latex.codecogs.com/gif.latex?[F,\hat{R}]" title="[F,\hat{R}]" />
3. B is computed from a convolution on the concatenation <img src="https://latex.codecogs.com/gif.latex?[F,\hat{R},\hat{S},O-\hat{R}\hat{S}]" title="[F,\hat{R},\hat{S},O-\hat{R}\hat{S}]" />

损失函数：

![](http://ww1.sinaimg.cn/large/006ocvumgy1g0f2p3m8jij30ed053wey.jpg)


- Inference
1. [论文阅读计划2(Deep Joint Rain Detection and Removal from a Single Image)](https://www.cnblogs.com/mengnan/p/9307532.html)
2. [图像去雨算法（基于卷积网络）](https://blog.csdn.net/u011692048/article/details/77743343)