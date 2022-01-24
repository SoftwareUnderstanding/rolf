# Variational Autoencoders

## Overview

### Autoencoders

![alt text](https://github.com/bsivanantham/VariationalAutoEncoder/blob/master/pastedImage0.png)

* Autoencoder takes input data it could be Image or vector  with high Dimensionality
* It gonna try and compress the data into a smaller representation it does this with two principal components is what we call encoder .
* From the Latent space with less dimension , the network will try to reconstruct the input by using again convolutional layer. 
* Loss function is computed by comparing input to output with the pixel difference.

### Variational Autoencoders
 
 ![alt text](https://github.com/bsivanantham/VariationalAutoEncoder/blob/master/pastedImage0%20(1).png)
 
 ```ruby
 
 generated_loss - mean(square(generated_image - real_image))
 latent_loss = KL_Divergence(latent_variable , unit_gaussian)
 loss = generation_loss +l atent_loss
 
 ```



## Dependency 

* pyDeepLearning
* Theano
* numpy
* scipy
Install dependencies using [pip](https://pip.pypa.io/en/stable/).

## Usage
This Git is intended as a playground for experimenting with various neural network models and libraries. It contains implementations of 
* mnist_mlp: A simple multilayer perceptron for MNIST implemented with keras
* mnist_cnn: A simple convolutional neural network for MNIST implemented with keras
* usps_cnn: A simple convolutional neural network for USPS dataset implemented with keras
* variational_autoencoder: Two implementations (one in pure Theano, one in lasagne) of the model proposed in 

*If cPickle import throughs error, change it to _pickle*

## Source
[Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
[goker erdogan](https://github.com/gokererdogan)


