Attentive-GAN论文笔记— Attentive Generative Adversarial Network for Raindrop Removal from A Single Image
=============================================

## Abstract&Introduction
附着在玻璃窗和相机镜头上的雨滴能一种损害背景图片的可视性并使图像降级。本文使用注意力机制的对抗生成网络来解决这一单图像去雨问题。

我们的主要想法是在生成和判别网络中引入注意力机制，这个注意力机制主要学习与地区与和其周围的情况。也就是说，生成网络会更多的注意雨滴所在区域及其周围，判别网络也能够评价重建区域的一致性。

## Raindrop Image Formation

我们将一幅经过雨滴降级的图片建模为背景图片和雨滴的结合体：

<img src="https://latex.codecogs.com/gif.latex?I=(1-M)\odot&space;B&plus;R" title="I=(1-M)\odot B+R" />

- I代表输入图片
- M代表二值掩模，M(x)=1时代表像素x位于雨区域，否则为背景图区域
- B代表背景图
- R代表雨滴带来的影响（背景信息的复杂混合、环境反射光和雨滴投射光）

我們可以將一張圖片(I)看為沒有水滴的部分以及有水滴的部分。有水滴的部分，透過Attention獲得能量強度（M），判斷是否為水滴。因此使用(1-M)來看看要拿多少能量的背景圖片，如果那個Pixel的水滴能量為0，那麼我們其實可以直接將那個Pixel視為沒有被水滴干擾的影像(B)，如果有被水滴影響(R)的話，那就要經由我們的Generator去做修正，最終希望獲得一張乾淨的圖片(B)。

## Raindrop Removal using Attention GAN

模型整体架构：

![](http://ww1.sinaimg.cn/large/006ocvumgy1g0f7tjaswfj310z0e079y.jpg)

Generative adversarial loss:

<img src="https://latex.codecogs.com/gif.latex?\underset{G}{min}\underset{D}{max}E_{R\sim&space;p_{clean}}[log(D(R))]&plus;E_{I\sim&space;p_{raindrop}}[log(1-D(G(I)))]" title="\underset{G}{min}\underset{D}{max}E_{R\sim p_{clean}}[log(D(R))]+E_{I\sim p_{raindrop}}[log(1-D(G(I)))]" />

整个网络同普通的GAN相同，主要分为Generator和Discriminator两个部分,Generator主要包含如下两个部分

- Attentive-Recurrent Network(Conv LSTM+Attention part)
    
    ![](http://ww1.sinaimg.cn/large/006ocvumgy1g0f93rym27j30i90afadg.jpg)
    
    Generator部分的视觉注意力模型主要用来定位一张图片里的目标区域从而抓住区域的特征，每一层的输入为上一层的Attention+Input Image,然后用5个residual block将特征提取出来，
    之后通过ConvLSTM以及一个ConvLayer制作出Attention Mask
    
    <img src="https://latex.codecogs.com/gif.latex?L_{ATT}(\{A\},M)=\sum^N_{t=1}\theta^{N-t}L_{MSE}(A_t,M)" title="L_{ATT}(\{A\},M)=\sum^N_{t=1}\theta^{N-t}L_{MSE}(A_t,M)" />
    
    文章中采用的是對每次輸出的Attention都做Mean squared error (MSE)

    這邊的設定是N=4， θ=0.8

    下圖可看出在越後面的Time step輸出的準確度越高。

    ![](http://ww1.sinaimg.cn/large/006ocvumgy1g0fc4maw2ij30sq083gtq.jpg)

- Contextual Autoencoder:

    ![](http://ww1.sinaimg.cn/large/006ocvumgy1g0fammp1pij30cs07jgms.jpg)

    輸入為最後輸出的Attention + Input Image

    autoencoder是由16個conv-relu blocks組成，以及有Skip connections被用來防止會有的模糊(blur)的輸出。(U-Net的形式)
    
    在這邊有兩個loss function : multi-scale loss 和 perceptual loss
    
    - Multi-Scale Loss
    
        <img src="https://latex.codecogs.com/gif.latex?L_{M}(\{S\},\{T\})=\sum^M_{t=1}\lambda_iL_{MSE}(S_i,T_i)" title="L_{M}(\{S\},\{T\})=\sum^M_{t=1}\lambda_iL_{MSE}(S_i,T_i)" />
        
        λ 設定為 0.6, 0.8, 1.0.
    
    - Perceptual Loss
    
        <img src="https://latex.codecogs.com/gif.latex?L_{P}(O,T)=L_{MSE}(VGG(O),VGG(T))" title="L_{P}(O,T)=L_{MSE}(VGG(O),VGG(T))" />
        
        这里的VGG是预训练好的CNN，从输入图中提取features。O这里是autoencoder的输出，T是ground-truth图片
        
综上，generator 的loss为：

<img src="https://latex.codecogs.com/gif.latex?L_{G}=0.01L_{GAN}(O)&plus;L_{ATT}({A},M)&plus;L_{M}(\{S\},\{T\})&plus;L_P(O,T)" title="L_{G}=0.01L_{GAN}(O)+L_{ATT}({A},M)+L_{M}(\{S\},\{T\})+L_P(O,T)" />
          
- Discriminator

    为了要辨别真假，Discriminator不只是要辨别水滴，还要确保整个图片看起来是正常的，因此需要对local（水滴处）和global（整体）这两个概念做检测。
    
    local
    
    如果我們知道哪邊是假的話，
    
    我們就可以針對某個區域做處理，
    
    但是對於移除水滴的問題來說，
    
    我們並不清楚哪些區域是有水滴的，
    
    因此換個想法，
    
    我們在Discriminator中能夠找到哪邊是水滴就好了。
    
    因此也在這邊採用Attention用於偵測水滴。
    
    值得一提的是雖然LSTM那邊的Convs為淺藍色，
    
    但是他的架構又和Discriminator的淺藍色Convs架構不同，
    
    目前的理解為淺藍色Convs代表Attention，
    
    他會透過loss function讓這個Attentive conv越來越像LSTM的Attentive conv
    
    <img src="https://latex.codecogs.com/gif.latex?L_{map}(O,R,A_N)=L_{MSE}(D_{map}(O),A_N)&plus;L_MSE(D_{map}(R),O)" title="L_{map}(O,R,A_N)=L_{MSE}(D_{map}(O),A_N)+L_MSE(D_{map}(R),O)" />
    
    Dmap指的是Discriminator的Attentive conv所产生的2D map
    
    Discriminator的整体的loss如下：
    
    <img src="https://latex.codecogs.com/gif.latex?L_{D}(O,R,A_N)=-log(D(R))-log(1-D(O))&plus;\gamma&space;L_{map}(O,R,A_N)" title="L_{D}(O,R,A_N)=-log(D(R))-log(1-D(O))+\gamma L_{map}(O,R,A_N)" />
    
    R ： 從training data sample出一張Groundtruth的圖片

    0 ： 指的是這張圖片是乾淨的，map偵測不到有水滴的部分，全填為0。
    
    γ ： set to 0.05
- Inference
 
[Attentive-GAN簡介 — Attentive Generative Adversarial Network for Raindrop Removal from A Single Image](https://medium.com/@xiaosean5408/attentive-gan%E7%B0%A1%E4%BB%8B-attentive-generative-adversarial-network-for-raindrop-removal-from-a-single-image-860ee597410f)