# Metal Image Recognition: Performing Image Recognition with Inception_v3 Network using Metal Performance Shaders Convolutional Neural Network routines

This sample demonstrates how to perform runtime inference for image recognition using a Convolutional Neural Network (CNN) built with Metal Performance Shaders. This sample is a port of the TensorFlow-trained Inception_v3 network, which was trained offline using the ImageNet dataset. The CNN creates, encodes, and submits different layers to the GPU. It then performs image recognition using trained parameters (weights and biases) that have been acquired and saved from the pre-trained network.

The Network can be found here:
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/image/imagenet/classify_image.py

Instructions to run this network on TensorFlow can be found here:
https://www.tensorflow.org/versions/r0.8/tutorials/image_recognition/index.html#image-recognition

The Original Network Paper can be found here:
http://arxiv.org/pdf/1512.00567v3.pdf

The network parameters are included in binary .dat files that are memory-mapped when needed.

The weights for this particular network were batch normalized but for inference the following may be used for every feature channel separately to get the corresponding weights and bias:

A = 𝛄 / √(s + 0.001), b = ß - ( A * m )

W = w*A

s: variance
m: mean
𝛄: gamma
ß: beta

w: weights of a feature channel
b: bias of a feature channel
W: batch nomalized weights

This is derived from:
https://arxiv.org/pdf/1502.03167v3.pdf

## Requirements

### Build

Xcode 8.0 or later; iOS 10.0 SDK or later

### Runtime

iOS 10.0 or later

### Device Feature Set

iOS GPU Family 2 v1
iOS GPU Family 2 v2
iOS GPU Family 3 v1

Copyright (C) 2016 Apple Inc. All rights reserved.
