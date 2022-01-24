# DCGAN-implementation using Pytorch.
DCGAN stands for deep convolutional Generative Adversarial Networks.

The code was implemented using the following paper  
  >https://arxiv.org/pdf/1511.06434.pdf


Input is a random normal vector “Code” that passes through de-convolution stacks and output an image.
Transposed convolutions are used in DCGAN'S to generate image from the random noise.During training the transposed convoltions weights are learned i.e that the model learned the distribution of the dataset.

The discriminator takes an image as input, passes through convolution stacks and output a probability (sigmoid value) telling whether or not the image is real.


