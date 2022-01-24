### Pytorch implementation of official DCGAN from paper [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)   
## Overview  
![Structure of DCGAN](https://miro.medium.com/max/1400/1*JaasxBD8iBGwd0ZNwC_Phw.png)
Generative Adversarial Networks are used to generate images that never existed before. They are able to learn about the world and create new versions of those images that never existed.   
They are divided into two basic components:  

A Generator — that creates the images.
A Discriminator — that assesses the images and tells the generator if they are similar to the trained images. These are based off real world examples.   

When training the network, both the generator and discriminator start from scratch and learn together.
The objective of a GAN is to train a data generator in order to imitate a given dataset. 
A GAN is similar to a zero sum game between two neural networks, the generator of data and a discriminator, trained to recognize original data from fakes created by the generator.   

### Check out my article [Deep Convolutional Generative Adversarial Network](https://medium.com/analytics-vidhya/deep-convolutional-generative-adversarial-network-4133bd4779ea) on Medium
