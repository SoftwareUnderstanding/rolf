# simple_fasterrcnn&maskrcnn_pytorch

![Version](https://img.shields.io/badge/version-0.0.1-brightgreen.svg "Version")
![License](https://img.shields.io/badge/License-MIT-orange.svg "License")
![OS](https://img.shields.io/badge/OS-windows%2Fmacos%2Flinux-blue.svg "OS")

This is a simplest implementation of fasterrcnn by pytorch when I learn the paper [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497). 
Now I also add maskrcnn [Mask R-CNN](https://arxiv.org/abs/1703.06870).
I give the key operation iou/nms/roi_pool/align_roi_pool in details by python and c++ , not just calling the torchvision library, so you are able to see the implementation of details. By the way, you can
compare the different implementation between mine and torchvision.opt library. I use The PASCAL Visual Object Classes(VOC2007) to train & test the faster rcnn model, the highest score is almost 0.687.And I use Ms Coco dataset(Coco2017) to train & test the mask rcnn model(I haven't finished it)

## Table of Contents

- [simple_fasterrcnn&maskrcnn_pytorch](#simple_fasterrcnnmaskrcnn_pytorch)
  - [Table of Contents](#table-of-contents)
  - [Background](#background)
  - [Requirements](#requirements)
  - [Install](#install)
  - [Usage](#usage)
  - [Scores](#scores)


## Background
This is a simplest implementation of fasterrcnn by pytorch when I learn the paper [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497). 
Now I also add maskrcnn [Mask R-CNN](https://arxiv.org/abs/1703.06870).
There are lots of implementation in github.But many of them is too complicated, or just calling the torchvision's module.
I implementation the key operation iou/nms/roi_pool/align_roi_pool in python & c++(not cuda), so you can see what the operation do on data.
What's more, the c++ version nms & roi_pool are written independently, and you can install them as a python package.

## Requirements
 ipdb>=0.13.5  
 matplotlib>=3.1.1  
 numpy>=1.17.5  
 Pillow>=8.0.1  
 scikit-image>=0.18.1  
 torch>=1.5.1  
 torchvision>=0.6.1  
 tqdm>=4.51.0  
 visdom>=0.1.8.9  
 pybind11>=2.6.2  

 The whole project has been test on python3.6

## Install
To install nms & roi_pool you should have [pybind11](https://github.com/pybind/pybind11/tree/stable) installed.

Then what you need to do is going into the **util** folder and run the following code:

```sh
pip install ./nms_mh
```
and  
```sh
pip install ./roi_pool_mh
```
and  
```sh
pip install ./align_roi_pool_mh
``` 

then you can use the library in other project not only in current one. You can find how to use it in file **test.py** which is under tests folder.
>Note:  
 nms&roi_pool&align_roi_pool is implemented in c++, so on different os platform(windows, mac os, linux) it has different compilation method. The pybind11 also has different installation instructions, so just follow the steps [here](https://pybind11.readthedocs.io/en/stable/installing.html)  

The following example is for using nms. And the nms_mh package also contains the iou & giou & ciou & diou functions.
```sh
import numpy as np
import nms_mh as m

rois = np.random.rand(12000,4).astype(np.float32)
scores = np.random.rand(12000).astype(np.float32)

keep_list = m.nms(rois, scores, 0.7)
```
or you can use different iou calculation algorithm by the parameter **iou_algo** of function **nms**, the default value for iou_algo is "iou", and it can be iou/giou/ciou/doiu.

And the roi_pool:

```sh
from torchvision.ops import RoIPool
import torch as t
import roi_pool_mh as mh

feat_x = t.rand(1, 2, 8, 8, requires_grad=True)
rois = t.tensor([[4,4,7,5], [1,3,3,7]], dtype=t.float32)

scale=1.0/2
roi_size=7
roi_pooling_lib = RoIPool((roi_size,roi_size),  scale)
roi_indices = t.zeros(rois.shape[0])
indices_and_rois = t.cat([roi_indices[:, None], rois], dim=1)
feat1 = roi_pooling_lib(feat_x, indices_and_rois)

roi_pooling = ROI_Pooling_C(roi_size, scale)
feat2 = roi_pooling.apply(feat_x, rois)

print(t.all(feat1==feat2))
assert(t.all(feat1==feat2))

# test backward
f1 = feat1.sum()
f1.backward()
grad1 = feat_x.grad.clone()

_ = feat_x.grad.zero_()
f2 = feat2.sum()
f2.backward()
grad2 = feat_x.grad.clone()

print(t.all(grad1==grad2))
assert(t.all(grad1==grad2))
```
Here ROI_Pooling_C is a wrapper class calling the roi_pool_mh's roi pool forward&backward function.You can have a look at the file test.py under the roi_pool_mh folder for details.  

## Usage
All you need to do is just run(just for fasterrcnn training):
```sh
python train_faster_rcnn.py
```
or for mask rcnn
```sh
python train_mask_rcnn.py
```

>## Note
>1.The parameters for training&testing is in file config.py which is under **config** folder, you can change any of them for testing.  
>2.Make sure the VOC2007 dataset is under **data** folder.But you can change the path by parameter **dataset_base_path** in config.py, then you can place the dataset files in any place as you like.

## Scores
The Score right now which I have achieved is a little above 0.687, which I use VGG16 as backbone.The weight is [here](https://pan.baidu.com/s/1TtznJQ98Y7JgaYv5IxNSeg)  (The baidu cloud storage, access code is k1gt)
Though I have added resnet family to the project, I haven't try it yet, and I will try it soon. By the way,you can use any other backbone, just have a look the folder **backbone** under **model** directory.When using different backbone, just remember to change the parameter **backbone** in file config.py.