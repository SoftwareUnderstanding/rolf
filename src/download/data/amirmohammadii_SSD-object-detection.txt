# SSD: Single Shot MultiBox Detector in TensorFlow

## Introduction
SSD is an unified framework for object detection with a single network. You can use the code to train/evaluate a network for object detection task. For more details, please refer to [arXiv paper](http://arxiv.org/abs/1512.02325).

This repository contains a TensorFlow re-implementation of the original [Caffe code](https://github.com/weiliu89/caffe/tree/ssd). At present, it only implements VGG-based SSD networks (with 300 and 512 inputs), but the architecture of the project is modular, and should make easy the implementation and training of other SSD variants (ResNet or Inception based for instance). Present TF checkpoints have been directly converted from SSD Caffe models.

The organisation is inspired by the [TF-Slim models](https://github.com/tensorflow/models/blob/master/research/inception/inception/slim/README.md) repository containing the implementation of popular architectures (ResNet, Inception and VGG).

## SSD minimal example

The SSD [SSD image detection](./ssd_image_detection.py) contains a minimal example of the SSD TensorFlow pipeline. Shortly, the detection is made of two main steps: running the SSD network on the image and post-processing the output using common algorithms.

To run the [SSD image detection](./ssd_image_detection.py) you first have to unzip the checkpoint files in ./checkpoint:

```
unzip ssd_300_vgg.ckpt.zip
```

Now run this:

```
python ssd_image_detection.py
```
