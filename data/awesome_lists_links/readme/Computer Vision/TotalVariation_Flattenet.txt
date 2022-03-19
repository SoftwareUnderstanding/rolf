# FlatteNet :sweat_smile: :sweat_smile: :sweat_smile:

## Introduction
This is a partially official pytorch implementation accompanying the publication 
[Flattenet: A Simple and Versatile Framework for Dense Pixelwise Prediction]( http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8932465&isnumber=8600701).
This code repo only contains the **semantic segmentation part**. It is worth noting that 
a number of **modifications** have been made after the paper has been published in order to 
improve the performance. We evaluate the adapted method on PASCAL-Context and PASCAL VOC 2012.

To deal with the reduced feature resolution problem, we introduce a novel Flattening Module 
which takes as input the coarse-grained feature maps (patch-wise visual descriptors) produced by 
a Fully Convolutional Network (FCN) and then outputs dense pixel-wise visual descriptors. 
The process described above is represented in the schematic diagram below. 
A FCN equipped with the Flattening Module, which we refer to as FlatteNet, can accomplish various 
dense prediction tasks in an effective and efficient manner. 
<p align="center">
<img src="figures/flattenmodule-1.png" width="800">
</p>

An illustration of the structure of Flattening Module is displayed below. We have newly incorporated a context
aggregation component into the design, which is implemented as a pyramid pooling module or self-attention
module.

<p align="center">
<img src="figures/gconv-1.png" width="100">
</p>

The overall architecture is displayed below.

<p align="center">
  <img src="figures/flattenet-1.png" width="200">
</p>

## Experimental Results
The training configuration files are included in *config* directory. 
All the models are trained on two NVIDIA GTX1080Ti GPUs with the official
Sync-BN.

*Note: We adopt the tweaks to ResNet architecture proposed* 
*in the [paper](https://arxiv.org/abs/1812.01187).*
*The weights of pretrained model are converted from* 
*[Gluon CV](https://github.com/dmlc/gluon-cv).*
### PASCAL-Context
The models are evaluated using six scales of 
[0.5; 0.75; 1; 1.25; 1.5; 1.75] and flipping. 

| Input size | Backbone | Context | mIoU(59 classes) | \#Params | GFLOPs |
| :--: | :--: | :--: | :--: | :--: | :--: |
| 480x480 | ResNet-101 | PPM | 53.3 | 49.7 | 40.0 |
| 512x512 | ResNet-101 | Self-Attention | 53.9 | 48.4 | 47.0|

*Note: The GFLOPs of self-attention block are not calculated.*

### PASCAL VOC
The models are evaluated using six scales of 
[0.5; 0.75; 1; 1.25; 1.5; 1.75] and flipping on the PASCAL VOC 2012 test set.
The input size is set to 512x512.

| COCO pretrain | Backbone | Context | mIoU | \#Params | GFLOPs |
| :--: | :--: | :--: | :--: | :--: | :--: |
| No | ResNet-101 | PPM | [83.09](http://host.robots.ox.ac.uk:8080/leaderboard/displaylb_main.php?challengeid=11&compid=5) | 49.7 | 45.4 |
| No | ResNet-101 | Self-Attention | [84.32](http://host.robots.ox.ac.uk:8080/leaderboard/displaylb_main.php?challengeid=11&compid=5) | 48.3 | 46.8|
| Yes | ResNet-101 | Self-Attention | [85.69](http://host.robots.ox.ac.uk:8080/leaderboard/displaylb_main.php?challengeid=11&compid=6) | 48.3 | 46.8|

*Note: The GFLOPs of self-attention block are not calculated.*

## Installation and Data Preparation
Pytorch version: 1.3.1

Please refer to [HRNet-Semantic-Segmentation ](https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/pytorch-v1.1)
for other details.
## Train and Test
For example, train FlatteNet on PASCAL-Context with a batch size of 8 on 2 GPUs:
```
python -m torch.distributed.launch --nproc_per_node=2 tools/train.py --cfg config/pctx_res101_att.yml
```
For example, evaluating our model on the PASCAL-Context validation set with multi-scale and flip testing:
```
python tools/test.py --cfg  config/pctx_res101_att.yml \
                     TEST.MODEL_FILE output/pascal_ctx/pctx_res101_att/best.pth
```
## Remarks
Despite the fact that the intention of this paper is to achieve my Master degree, 
I sincerely hope this work or code would be helpful for your research. If you 
have any problems, feel free to open an issue.

Moreover, I could not test our method on other datasets/benchmarks due to the 
limited access to computational resources. :disappointed: :disappointed: :disappointed:

## Note
Due to the limited effort, this repo will not be maintained. For those interested in this work, please follow the configuration files available in the *config* folder to train your own models. The pre-trained models will no longer be available. I apologize for any inconvenience caused and thank you for your interest.

## Acknowledgement
The code has been largely taken from [HRNet-Semantic-Segmentation ](https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/pytorch-v1.1).

Other References:

[PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding)
