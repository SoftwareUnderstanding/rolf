# U-NET reference implementation in MXNET

Reference implementation of [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
## Using a Python VirtualEnv environment with VSCode

Create a symbolic link to a `env` folder in the root of the project that point to your MXNET VirtualEnv

```sh
   ln -s  ~/environments/mxnet1.7/ ~/dev/unet-reference-impl/env
```

## Dataset

There are multiple datasets provided for testing the implementation.
They have been downloaded from sources bellow and extracted into `dataset` folder.

* [ISBI Challenge: Segmentation of neuronal structures in EM stacks](http://brainiac2.mit.edu/isbi_challenge/)

Data folder layout

* train images + segmentation masks
* validation images + segmentation masks
* test images + segmentation masks

```sh
├── dataset
    ├── train
    │   ├── image
    │   │   ├── 1.png
    │   │   └── 2.png
    │   └── mask
    │       ├── 1.png
    │       └── 2.png
    ├── validate
    │   ├── image
    │   │   ├── 1.png
    │   │   └── 2.png
    │   └── mask
    │       ├── 1.png
    │       └── 2.png
    └── test
        ├── image
        │   ├── 1.png
        │   └── 2.png
        └── mask
            ├── 1.png
            └── 2.png
```

## Usage


### Training

```sh
python ./segmenter.py --checkpoint-file ./unet_best.params
```

### Restarting training

```sh
python ./segmenter.py --checkpoint=load --checkpoint-file ./unet_best.params
```

### Evaluating model

```sh
python ./evaluate.py --image=./input.png ----network-param./unet_best.params
```


## U-Net network features

* Resnet Block
  * Added Residual/Skip connection (ResBlock) as the original paper did not include them
* Batch Norm / Layer Norm
  * Added normalization layer
* Dropout
  * Added support for optional dropout layer
* Conv2DTranspose / UpSample
  * Added support to switche between `Conv2DTranspose` and `UpSampling` http://distill.pub/2016/deconv-checkerboard/
  
## Tensorboard

 ```sh
 tensorboard --logdir ./logs
 ``` 

## Dependencies 

* MXNET >= 1.7 

```sh
python -m pip install git+https://github.com/aleju/imgaug
python -m pip install tensorboard
python -m pip install mxboard
```



## Model Information

```sh
    # NCHW (N:batch_size, C:channel, H:height, W:width)
    net.summary(nd.ones((1, 3, 512, 512)))  
```

```sh
--------------------------------------------------------------------------------
        Layer (type)                                Output Shape         Param #
================================================================================
               Input                            (1, 3, 512, 512)               0
            Conv2D-1                           (1, 64, 512, 512)             256
            Conv2D-2                           (1, 64, 512, 512)            1792
         LayerNorm-3                           (1, 64, 512, 512)            1024
            Conv2D-4                           (1, 64, 512, 512)           36928
         LayerNorm-5                           (1, 64, 512, 512)            1024
     BaseConvBlock-6                           (1, 64, 512, 512)               0
         MaxPool2D-7                           (1, 64, 256, 256)               0
            Conv2D-8                          (1, 128, 256, 256)            8320
            Conv2D-9                          (1, 128, 256, 256)           73856
        LayerNorm-10                          (1, 128, 256, 256)             512
           Conv2D-11                          (1, 128, 256, 256)          147584
        LayerNorm-12                          (1, 128, 256, 256)             512
    BaseConvBlock-13                          (1, 128, 256, 256)               0
  DownSampleBlock-14                          (1, 128, 256, 256)               0
        MaxPool2D-15                          (1, 128, 128, 128)               0
           Conv2D-16                          (1, 256, 128, 128)           33024
           Conv2D-17                          (1, 256, 128, 128)          295168
        LayerNorm-18                          (1, 256, 128, 128)             256
           Conv2D-19                          (1, 256, 128, 128)          590080
        LayerNorm-20                          (1, 256, 128, 128)             256
    BaseConvBlock-21                          (1, 256, 128, 128)               0
  DownSampleBlock-22                          (1, 256, 128, 128)               0
        MaxPool2D-23                            (1, 256, 64, 64)               0
           Conv2D-24                            (1, 512, 64, 64)          131584
           Conv2D-25                            (1, 512, 64, 64)         1180160
        LayerNorm-26                            (1, 512, 64, 64)             128
           Conv2D-27                            (1, 512, 64, 64)         2359808
        LayerNorm-28                            (1, 512, 64, 64)             128
    BaseConvBlock-29                            (1, 512, 64, 64)               0
  DownSampleBlock-30                            (1, 512, 64, 64)               0
        MaxPool2D-31                            (1, 512, 32, 32)               0
           Conv2D-32                           (1, 1024, 32, 32)          525312
           Conv2D-33                           (1, 1024, 32, 32)         4719616
        LayerNorm-34                           (1, 1024, 32, 32)              64
           Conv2D-35                           (1, 1024, 32, 32)         9438208
        LayerNorm-36                           (1, 1024, 32, 32)              64
    BaseConvBlock-37                           (1, 1024, 32, 32)               0
  DownSampleBlock-38                           (1, 1024, 32, 32)               0
           Conv2D-39                            (1, 512, 64, 64)         4719104
UpsampleConvLayer-40                            (1, 512, 64, 64)               0
           Conv2D-41                            (1, 512, 64, 64)          524800
           Conv2D-42                            (1, 512, 64, 64)         4719104
        LayerNorm-43                            (1, 512, 64, 64)             128
           Conv2D-44                            (1, 512, 64, 64)         2359808
        LayerNorm-45                            (1, 512, 64, 64)             128
    BaseConvBlock-46                            (1, 512, 64, 64)               0
           Conv2D-47                          (1, 256, 128, 128)         1179904
UpsampleConvLayer-48                          (1, 256, 128, 128)               0
           Conv2D-49                          (1, 256, 128, 128)          131328
           Conv2D-50                          (1, 256, 128, 128)         1179904
        LayerNorm-51                          (1, 256, 128, 128)             256
           Conv2D-52                          (1, 256, 128, 128)          590080
        LayerNorm-53                          (1, 256, 128, 128)             256
    BaseConvBlock-54                          (1, 256, 128, 128)               0
           Conv2D-55                          (1, 128, 256, 256)          295040
UpsampleConvLayer-56                          (1, 128, 256, 256)               0
           Conv2D-57                          (1, 128, 256, 256)           32896
           Conv2D-58                          (1, 128, 256, 256)          295040
        LayerNorm-59                          (1, 128, 256, 256)             512
           Conv2D-60                          (1, 128, 256, 256)          147584
        LayerNorm-61                          (1, 128, 256, 256)             512
    BaseConvBlock-62                          (1, 128, 256, 256)               0
           Conv2D-63                           (1, 64, 512, 512)           73792
UpsampleConvLayer-64                           (1, 64, 512, 512)               0
           Conv2D-65                           (1, 64, 512, 512)            8256
           Conv2D-66                           (1, 64, 512, 512)           73792
        LayerNorm-67                           (1, 64, 512, 512)            1024
           Conv2D-68                           (1, 64, 512, 512)           36928
        LayerNorm-69                           (1, 64, 512, 512)            1024
    BaseConvBlock-70                           (1, 64, 512, 512)               0
           Conv2D-71                            (1, 2, 512, 512)             130
================================================================================
Parameters in forward computation graph, duplicate included
   Total params: 35916994
   Trainable params: 35916994
   Non-trainable params: 0
Shared params in forward computation graph: 0
Unique parameters in model: 35916994
--------------------------------------------------------------------------------

```



## Notes

https://towardsdatascience.com/counting-no-of-parameters-in-deep-learning-models-by-hand-8f1716241889