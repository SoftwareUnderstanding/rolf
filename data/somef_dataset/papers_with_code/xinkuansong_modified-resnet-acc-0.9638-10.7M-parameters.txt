# Modified Residual Network

By Song xinkuan.

### Table of Contents
0. [Introduction](#introduction)
0. [Model Discription, Implementation, and Training Detatils](#model-discription-implementation-and-training-details)
0. [Results](#results)

### Introduction

This repository contains the original modified residual network which modified the classical resnet "Deep Residual Learning for Image Recognition" (http://arxiv.org/abs/1512.03385). Original residual block was modified to improve model performance. 

### Model Discription and Implementation

#### Model disciption
Firstly, the modified residual network introduces filtering mechanism in the basic residual block. Comparing to original residual block, shortcut connection was introduced just after the relu activation and an additional batch normalization was added after the shortcut connetction. See in the following image.
![Residual block Vs modified residual block](https://github.com/xinkuansong/modified-resnet-acc-0.9638-10.7M-parameters/blob/master/images/residual%20block%20VS%20modified%20residual%20block.PNG)
The shortcut connection between bn of former block and relu activation of current block functioned as filtering mechanism, it can determine which part of features should be emphasized and which part of features should not to be. Detail illustration of modified residual block below. 
![Modified residual block](https://github.com/xinkuansong/modified-resnet-acc-0.9638-10.7M-parameters/blob/master/images/modified%20residual%20block.PNG)
Secondly, no bottleneck architecture and no doubling number of feature maps after each time shrinking feature map size in modified residual block.

#### Model implementation and training details

Model implemented in keras with tensorflow backend.  
Depth: 76  
Number of parameters: 10.73M  
Datasets: cifar10+  
Learning rate schedule: the training process of modified residual network was separated into two stages, including stepwise decay of stage1(240 epochs) and consine decay of stage2(150 epochs)  
Equipment: Single NVDIA 1080Ti  
### Results
Learning rate schedule of two training stages:
![Learning rate schedule](https://github.com/xinkuansong/modified-resnet-acc-0.9638-10.7M-parameters/blob/master/images/lr_log.png)
Training and validation loss: 
![Training and validation loss](https://github.com/xinkuansong/modified-resnet-acc-0.9638-10.7M-parameters/blob/master/images/loss.png)
Training and validation accuracy: 
![Training and validation accuracy](https://github.com/xinkuansong/modified-resnet-acc-0.9638-10.7M-parameters/blob/master/images/acc.png)
The final validation accuracy is 96.38%. (Three times maximum: 96.44%, mean: 96.34%)

### Connection:
You can connect me through: sxk_ml@163.com
