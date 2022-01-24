# Extended caffe 

### Introduction
This repository contains an extended caffe wich is modified from caffe version of [yjxiong](https://github.com/yjxiong/caffe/tree/mem) and introduces many new features.   

### Features
-  on-the-fly data augmentation, which is used in **ImageSegData** layer, including mirror, crop, scale, smooth filer, rotation, translation, please refers to **caffe\src\caffe\data_transformer\TransformImgAndSeg2**
an example is as follows:
```
layer {
  name: "data"
  type: "ImageSegData"
  top: "data"
  top: "label"
  top: "data_dim"
  include {
    phase: TRAIN
  }
  transform_param {
    mirror: true
    crop_size: 352
    mean_value: 104.008
    mean_value: 116.669
    mean_value: 122.675
    scale_factors: 0.5
    scale_factors: 0.75
    scale_factors: 1
    scale_factors: 1.25
    scale_factors: 1.5
    scale_factors: 1.75
    scale_factors: 2.0
	smooth_filtering: true
	max_smooth: 6
	apply_probability: 0.5
	max_rotation_angle: 60
	max_translation: 30
  }
  image_data_param {
    root_folder: "/data1/caiyong.wang/data/Point/CASIA/"
    source: "/data1/caiyong.wang/data/Point/CASIA/list/train_edge.txt"
    batch_size: 1
    shuffle: true
    label_type: PIXEL
  }
}
```
- include interp_layer used in deeplab
see  [http://liangchiehchen.com/projects/DeepLab.html](http://liangchiehchen.com/projects/DeepLab.html)
- include balance_cross_entropy_loss_layer used in hed
see [Holistically-Nested Edge Detection](https://github.com/s9xie/hed)
- include normalize_layer: L2 normalization for parsenet
see [ParseNet: Looking Wider to See Better](https://arxiv.org/abs/1506.04579)
- support pooling with **bin size, output_size** for [pspnet](https://github.com/hszhao/PSPNet),[segnet](http://mi.eng.cam.ac.uk/projects/segnet/)
- support upsample_layer used in segnet
````
layer {
  name: "pool4"
  type: "Pooling"
  bottom: "conv4_3"
  top: "pool4"
  top: "pool4_idx"
  top: "pool4_size"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
	output_size: true
  }
}
layer {
  name: "upsample4"
  type: "Upsample"
  bottom: "conv5_1_D"
  bottom: "pool4_idx"
  bottom: "pool4_size"
  top: "pool4_D"
}
```` 
- include dice_loss_layer
- include focal_sigmoid_loss_layer, the usage is simlar with **SigmoidCrossEntropyLoss** 
```
layer {
  name: "loss_mask"
  type: "FocalSigmoidLoss"
  bottom: "mask_pred"
  bottom: "mask_label"
  top: "loss_mask"
  loss_weight: 10
  loss_param {
       ignore_label: 255
	   normalize: true
  }
  focal_sigmoid_loss_param
  {
	  alpha: 0.95
      gamma: 2
  }
}
```
- include focal_softmax_loss_layer, modified from [https://github.com/chuanqi305/FocalLoss](https://github.com/chuanqi305/FocalLoss), the usage is similar with 
**SoftmaxWithLoss**, more details, please see [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
- include prelu_layer
- include smooth_L1_loss_layer
- include selu_layer
- support deconvolution upsampling type: nearest
````
layer {
  name: "out_2_up4"
  type: "Deconvolution"
  bottom: "out_2"
  top: "out_2_up4"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  convolution_param {
    num_output: 2
    bias_term: false
    pad: 0
    kernel_size: 4
    group: 2
    stride: 4
    weight_filler {
      type: "nearest"
    }
  }
}
````
- include my_spp_layer  for  spatial pyramid pooling
see [SPPNet：Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition](https://arxiv.org/abs/1406.4729)
```
layer {
  name: "spatial_pyramid_pooling"
  type: "MySPP"
  bottom: "conv5"
  top: "pool5"
  my_spp_param {
    pool: MAX
    bin_size: 2  
	bin_size: 3 
    bin_size: 6 
  }
} 
````
### Installation 

For installation, please follow the instructions of [Caffe](https://github.com/BVLC/caffe).
For chinese users, please refers to [https://blog.csdn.net/xiamentingtao/article/details/78283336](https://blog.csdn.net/xiamentingtao/article/details/78283336),
 [https://blog.csdn.net/xiamentingtao/article/details/78266153](https://blog.csdn.net/xiamentingtao/article/details/78266153) and [https://wangcaiyong.blog.csdn.net/article/details/110262549](https://wangcaiyong.blog.csdn.net/article/details/110262549).

 To enable cuDNN for GPU acceleration, cuDNN v6 is needed. 

The code has been tested successfully on CentOS 6.9  with CUDA 8.0.

### Questions
Please contact wangcaiyong2017@ia.ac.cn

----
Following is the original README of Caffe.

# Caffe

Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by the Berkeley Vision and Learning Center ([BVLC](http://bvlc.eecs.berkeley.edu)) and community contributors.

Check out the [project site](http://caffe.berkeleyvision.org) for all the details like

- [DIY Deep Learning for Vision with Caffe](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.p)
- [Tutorial Documentation](http://caffe.berkeleyvision.org/tutorial/)
- [BVLC reference models](http://caffe.berkeleyvision.org/model_zoo.html) and the [community model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
- [Installation instructions](http://caffe.berkeleyvision.org/installation.html)

and step-by-step examples.

[![Join the chat at https://gitter.im/BVLC/caffe](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/BVLC/caffe?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Please join the [caffe-users group](https://groups.google.com/forum/#!forum/caffe-users) or [gitter chat](https://gitter.im/BVLC/caffe) to ask questions and talk about methods and models.
Framework development discussions and thorough bug reports are collected on [Issues](https://github.com/BVLC/caffe/issues).

Happy brewing!

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
