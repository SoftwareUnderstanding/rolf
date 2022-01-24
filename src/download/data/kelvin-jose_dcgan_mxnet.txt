# DCGAN

## What is DCGAN?

A DCGAN is a direct extension of the GAN which uses convolutional and convolutional-transpose layers in the discriminator and generator, respectively. It was first described by Radford et. al. in the paper [Unsupervised Representation Learning With Deep Convolutional Generative Adversarial Networks](https://arxiv.org/pdf/1511.06434.pdf). 

The discriminator is made up of **strided convolution**, **batch norm** layers, and **LeakyReLU** activations. The input is a 3x64x64 input image and the output is a scalar probability that the input is from the real data distribution. The generator is comprised of **convolutional-transpose** layers, batch norm layers, and **ReLU** activations. The input is a latent vector, z, that is drawn from a standard normal distribution and the output is a 3x64x64 RGB image. The strided conv-transpose layers allow the latent vector to be transformed into a volume with the same shape as an image. In the paper, the authors also give some tips about how to setup the optimizers, how to calculate the loss functions, and how to initialize the model weights, all of which will be explained in the coming sections.

![DCGAN architecture](https://gluon.mxnet.io/_images/dcgan.png)

## Technologies Used

 - Mxnet
 - Python

## Reference

 - https://arxiv.org/pdf/1511.06434.pdf


