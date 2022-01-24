# Spatial Transform Layer

The project implements Spatial Transform Layer with tf.keras API for Tensorflow 2.0. 

## Introduction

Spatial Transform layer is an algorithm proposed in paper ("Spatial Transformer Networks")[https://arxiv.org/abs/1506.02025]. Spatial Transform Layer can enhance model's invariance to translation, scale, rotation and more generic warping. It usually serves as the first layer of neural network to preprocess input image with affine transformation to align object in the input image to the training samples automatically. 

## Test the Layers

The project provides two tf.keras layers. 

AffineLayer affine transform the input batch of images according to a batch of affine matrices. You can test the AffineLayer by executing.

```Base
python3 AffineLayer.py
```

The demo will show the rotation result of the input image.

SpatialTransformLayer align object automatically during training.


