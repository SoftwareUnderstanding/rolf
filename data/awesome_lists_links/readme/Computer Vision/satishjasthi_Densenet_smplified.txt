# Description

This is a quick implementation of the DenseNet model described in the paper *"Densely Connected Convolutional Networks"* by Huang et al. ([arXiv](https://arxiv.org/abs/1608.06993))

It has only been tested on the Cifar-10 dataset without data augmentation, but it should work fine on any dataset.

# Getting started
The basic model is defined in *DenseNet.py*.

The scipt *cifar10_densenet_classification.py* provides an example on how to create and use the model on Cifar-10 classification.

Finally, *utils.py* contains a few helper functions.

## Prerequisites
* Keras (>= 2) (only tested with Tensorflow backend)
* numpy (>= 1.13)

# Results
Below are the results of running both DenseNet and DenseNet-BC models on Cifar-10 dataset with the same hyperparameters and optimization techniques as in the original paper.

## DenseNet (L=40, k=12)
![DenseNet_loss](/results/DenseNet_loss.png)
![DenseNet_accuracy](/results/DenseNet_accuracy.png)

## DenseNet-BC (L=100, k=12)
![DenseNet-BC_loss](/results/DenseNet-BC_loss.png)
![DenseNet-BC_accuracy](/results/DenseNet-BC_accuracy.png)

# TODO
* Add data augmentation techniques
* Try different architectures
* Try other optimizers, eg Adam
* Try out transfer learning on ImageNet
