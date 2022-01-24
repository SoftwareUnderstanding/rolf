# DeepLearningLab5

## Project Proposal: Convolutional and Deep Learning Networks on CIFAR-10

### Jonas Wechsler, Nicholas Jang, Lars Traaholt Vågnes

For our project, we are going to train a convolutional neural net with the CIFAR-10 data set. We are going to try several architectures based off of ones described in publications found on Arxiv (see below). Examples include alternating convolution and max-pooling layers followed a small number of fully connected layers and a single convolutional layer with increased stride followed by the same fully connected layers. We are going to pre-train both with an autoencoder or RBM, as well as a method described by Dmytro Mishkin and Jiri Matas in All You Need is a Good Init, in which you pre-initialize the weights of each convolutional layer with orthonormal matrices, and then sequentially normalize the variance of each layer's output to be equal to one. We'd like to potentially re-create these nets and verify their accuracy, though due to time restrictions we're not yet sure if that's fully possible with our available hardware. We plan to use GPU accelerated Tensorflow using Python. After these initial tests, we would like to experiment with different architectures to achieve the highest performance. We will measure the success of our project if we can achieve higher test accuracies on the data than methods discussed in class. 

## Project Outline

https://www.tensorflow.org/tutorials/deep_cnn

http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html

* Use tensorflow
* GPU accelerated
* Pretrain with autoencoder or RBM
* CIFAR-10
* Max pooling?
* Try some stuff from papers
* https://arxiv.org/abs/1412.6071
* https://arxiv.org/pdf/1412.6806.pdf
    * Alternating convolution and max-pooling layers followed by a small number of fully connected layers. 
    * max-pooling can simply be replaced by a convolutional layer with increased stride without loss in accuracy on several image recognition benchmarks.
* https://arxiv.org/abs/1511.06422
    * First, pre-initialize weights of each convolution or inner-product layer with orthonormal matrices. Second, proceed from the first to the final layer, normalizing the variance of the output of each layer to be equal to one

## Requirements

This repo uses [Tensorflow](https://www.tensorflow.org/install/install_windows).

## Running

The main file for this repo is allconv.py
