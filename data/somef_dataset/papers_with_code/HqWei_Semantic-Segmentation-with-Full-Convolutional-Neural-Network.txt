# Semantic-Segmentation-with-Full-Convolutional-Neural-Network

## Introduction
Semantic segmentation in Weizmann horse dataset and Labeled Face in the Wild dataset.
## Method

### 最早的全卷积语义分割网络：

https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf

### 目前比较热门的结构：
#### PSPNet ：Pyramid Scene Parsing Network
https://github.com/hszhao/PSPNet

#### DeeplabV3 ：Rethinking Atrous Convolution for Semantic Image Segmentation

https://arxiv.org/abs/1706.05587

https://github.com/NanqingD/DeepLabV3-Tensorflow

#### 基于attention机制的：
##### CCNet： Criss-Cross Attention for Semantic Segmentation
https://github.com/speedinghzl/CCNet

##### DAN ： Dual Attention Network for Scene Segmentation
https://github.com/junfu1115/DANet

## Data

The download link for Weizmann horse dataset:

http://www.msri.org/people/members/eranb/

Labeled Face in the Wild:

http://vis-www.cs.umass.edu/lfw/
http://vis-www.cs.umass.edu/lfw/part_labels/

Related semantic datasets:

https://blog.csdn.net/bevison/article/details/78123403

## 数据转换
python Semantic_segmentation_lfw/data_process.py

## Result:
python ./imgs/ 	vis_result.py

![image](https://github.com/HqWei/Semantic-Segmentation-with-FCN/blob/master/imgs/result.jpg)
