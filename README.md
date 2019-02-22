Deep Convolutional Network in Single Image Derain
==================================

## Todo List(02.21)
    - Solution to loss bump
    - fine tune the super parameter

## Overview
Restoring rain images is important for many computer vision applications in outdoor scenes since rain streaks can severely degrade the visibility causing many current computer vision algorithms fails to work.

In recent years, various methods mainly based on CNNs have been proposed to address this problem.

This repo records some papers with implementation and finally make comparison between them.

## Important models' implementations in pytorch

### Dependencies
- Python>=3.6
- Pytorch>=1.0.0
- Opencv>=3.1.0
- visdom

```bash
    pip install -r requirements.txt
``` 

### Project Structure
- checkpoints : holds checkpoints
- datasets : 
- losses: losses like ssim and psnr
- models: holds model for training or testing
- pretrained_models: holds pretrained model for fine-ture
- settings : holds seperate setting config for model
- utils : practical tools 

### Datasets
- RESCAN : [Rain800](https://drive.google.com/drive/folders/0Bw2e6Q0nQQvGbi1xV1Yxd09rY2s)
- JORDER : [Rain100H,Rain100L](http://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html)
- DID-MDN : [DID-MDN](https://drive.google.com/file/d/1cMXWICiblTsRl1zjN8FizF5hXOpVOJz4/view?usp=sharing)
### Training

Two parameters:

-c, --config : each networks configuration files which contains the parameter about model and training details.

-t, --type : denote the model structure fully connected network or generative adversarial networks

Example usage:

```bash
    python train.py -c ./configs/didmdn_didmdn_rain.yaml -t 'fcn'
``` 
### Testing

In order to test, you need to run the following command and set input_path to the folder with images (optionally, also set img_list to a list with subset of these image names), specify scaling by setting image_size (required for CelebA-HQ), file with network weights (net_path) and output directory (output_path).

Example usage:

```bash
python test.py  -c ./configs/didmdn_didmdn_rain.yaml -t 'fcn'
```

### Pretrained models


### Result(Todo)



## Important Papers

1. DID-MDN(<u>*SOTA*</u>)

    - Paper
    
         [Density-aware Single Image De-raining using a Multi-stream Dense Network, He Zhang, Vishal M. Patel, 2018](https://arxiv.org/pdf/1802.07412.pdf)
    
    - Github
    
        [DID-MDN](https://github.com/hezhangsprinter/DID-MDN)
        
    - [Notes](./Notes/DID-MDN.md)
2. Current JORDAR

    - Paper
    
         [Deep Joint Rain Detection and Removal from a Single Image, Wenhan Yang, Robby T. Tan et al, 2017](https://arxiv.org/pdf/1609.07769.pdf)
    
    - Github
    
        [Joint Rain Detection and Removal from a Single Image](https://github.com/ZhangXinNan/RainDetectionAndRemoval)
        
    - [Notes](./Notes/CJORDAR.md) 

3. RESCAN

    - Paper
    
         [Recurrent Squeeze-and-Excitation Context Aggregation Net for Single Image Deraining, Xia Li, Jianlong Wu et al, 2018](https://arxiv.org/pdf/1807.05698.pdf)
    
    - Github
    
        [RESCAN: Recurrent Squeeze-and-Excitation Context Aggregation Net](https://github.com/XiaLiPKU/RESCAN)
        
    - [Notes](./Notes/RESCAN.md) 

4. Attentive-GAN for Raindrop Removal

    - Paper
        
        [Attentive Generative Adversarial Network for Raindrop Removal from A Single Image,Rui Qian, Robby T. Tan et al, 2017](https://arxiv.org/pdf/1711.10098.pdf)
        
    - Github
    
        [Attentive Generative Adversarial Network for Raindrop Removal from A Single Image (CVPR'2018)](https://github.com/rui1996/DeRaindrop)
        
    - [Notes](./Notes/Att-GAN.md)
5. Perceptual-GAN

    - Paper
        
        [Perceptual Adversarial Networks for Image-to-Image Transformation,Chaoyue Wang, Chang Xu, et al, 2017](https://arxiv.org/pdf/1706.09138.pdf)
        
    - [Notes](./Notes/P_GAN.md)

    
    
