# DCGAN_pytorch
An implementation of Deep Convolutional Generative Adversarial Network, or DCGAN for short.

Paper: https://arxiv.org/pdf/1511.06434.pdf

The DCGAN is a fully convolutional network with all feed-forward layers removed. Each layer in the Discriminator is a convolutional layer and the Discriminator downsizes the image fed through it via strided convolutions. Strided convolutions are preferred over pooling as strided convolutions are operations that the network can tune via learning, making them optimal for the DCGAN. The generator takes in a latent noise vector and applies a convolutional tranpose operation (sometimes referred to as a deconvolution) to upsample the latent noise vector into the correct dimension image.

Here are the architecture diagrams given in the paper:

## Generator

![](data/uploads/generator.png)

## Discriminator

![](data/uploads/discriminator.png)

The DCGAN is able to more fluidly learn dynamic images compared to a vanilla feed-forward GAN producing better results. Some examples show quit pleasing going through the batch of generated fake images.

![](data/saved_images/epoch_15_checkpoint.jpg)

The DCGAN is non-conditioned meaning you are unable to choose exact qualities desired from the generated output data. For a Conditional-DCGAN please refer my 'Conditional-DCGAN' repository: https://github.com/u7javed/Conditional-DCGAN

## Epoch Progression of DCGAN

![](data/saved_images/epoch_progression.gif)
