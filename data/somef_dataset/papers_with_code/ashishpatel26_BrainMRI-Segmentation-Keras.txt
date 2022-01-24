# MRI Brain Segmentation

## Introduction

This project is to study the use of Convolutional Neural Network and in particular the ResNet architecture. The show case is segmentation of Magnetic Resonance Images (MRI) of human brain into anatomical regions[2]. We will extend the ResNet topology into the processing of 3-dimensional voxels.


## Datasets, Preprocessing and Sampling
The Brain MRI images used are provided by the Medical Image Computing and Computer Assisted Intervention Society (MICCAI) in their 15th International Conference in 2012. The images are captured from 20 patients. 15 of them will be used for training while the rest serve for testing.

The Brain MRI of each patient is a 3-dimensional voxel datasets. A single image is 256x256 in size while the depth varies between patients. The datasets include labels which classify 135 different sections of the brain.

We will randomly sample a small voxel as input the the neural network. Since not all pixels in the datasets contain meaningful data and some useful sub-image may be placed near boundary of the dataset, zero-padding is added to the boundary of the dataset at first. Then the dataset will be normalized and equalized. During training and testing, the sampling processing will filter the zero-valued pixel so only voxel that contain meaningful data will be fed into the network.

## Implementation

Using Keras, we implemented a function which return a Keras model object of residual block and by calling it multiple times with desired parameters (e.g. no. of filters to use, transformation of volume between input and output), we formed a network which only takes in a 26x26x26 voxel of MRI images.

This size of voxel can carry 8 times more of information than the original 13x13x13 voxel in [2]. We have experimented our network with 2D patch input and found that it doesn’t contribute much to the accuracy so in this we discarded it to focus on training a 3D voxel residual network. We particularly employed the full pre‐activation version of residual block presented in [3], where Kaiming He found that such version offers much better training performance than the original ResNet.

Keras implementation of single Residual Block:
![Implementation of ResNet-V2](../master/doc/images/resnet_diagram.png)


## Results

Training accuracy:
![Training](../master/doc/images/train_acc.png)

Validation accuracy:
![Validation](../master/doc/images/train_val_acc.png)

## Demo

Source Image:

![Training](../master/doc/images/MRI_src.png)

Ground Truth:

![Training](../master/doc/images/MRI_truth.png)

Our Prediction:

![Training](../master/doc/images/MRI_pred.png)

Difference:

![Training](../master/doc/images/MRI_diff.png)


## Reference

[1] Deep Residual Learning for Image Recognition

https://arxiv.org/abs/1512.03385#

[2] Deep Neural Netowrk for Anatomical Brain Segmentation

https://arxiv.org/abs/1502.02445

[3] Identity Mappings in Deep Residual Networkss

https://arxiv.org/abs/1603.05027

[4] The Medical Image Computing and Computer Assisted Intervention Society
http://www.miccai.org/




