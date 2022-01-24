# DCGAN
Generation of Fake images using Pytorch
## Introduction

Generative Adversarial Networks (GANs) are one of the most interesting ideas in computer science today. Here two models named Generator and Discriminator are trained simultaneously. As the name says Generator generates the fake images or we can say it generates a random noise and the Discriminator job is to classify whether the image is fake or not. Here the only job of Generator is to fake the Discriminator. In this project we are using DCGAN(Deep Convolutional Generative Adversarial Network). A DCGAN is a direct extension of the GAN described above, except that it explicitly uses convolutional and convolutional-transpose layers in the discriminator and generator, respectively. DCGANs actually comes under Unsupervised Learning and was first described by Radford et. al. in the paper Unsupervised Representation Learning With Deep Convolutional Generative Adversarial Networks.

![](https://miro.medium.com/max/2850/1*Mw2c3eY5khtXafe5W-Ms_w.jpeg)

## Generator

1. First we have to provide a random noise to the generator as a input.
2. Make sure that the Generator never ever sees the real images.
3. The only job of generator is to generate fake images and to fool the discriminator. It gives the fake images to the discriminator and says "Hey, these are real Images!" to the    discriminator.
4. This is accomplished through a series of strided two dimensional convolutional transpose layers, each paired with a 2d batch norm layer and a relu activation.
5. The output of the generator is fed through a tanh function to return it to the input data range of [-1,1].

## Discriminator

1. For the discriminator there will be tow inputs, one from Generator(fake images) and the other is real images that should be given by us.
2. It classifies whether the given image is of real face or not.
3. Most probably for the first time the discriminator classifies all the images(random noise) from generator as fake images, because it is a random noise.
4. The network looks quite opposite to the discriminator.
5. As mentioned, the discriminator is a binary classification network that takes an image as input and outputs a scalar probability that the input image is real (as opposed to  	fake).
6. Here, Discriminator takes a 3x64x64 input image, processes it through a series of Conv2d, BatchNorm2d, and LeakyReLU layers, and outputs the final probability through a          Sigmoid activation function. 

![](https://gluon.mxnet.io/_images/dcgan.png)

## Loss Function

![](https://cdn-images-1.medium.com/max/1600/1*vh9PN7ktJMs7FH71yAnKKg.png)

Here,
   m is number of samples,
   D(x) is Discriminator where x(i) is training examples,
   G(z) is Generator where z(i) is latent vector or random noise.
   
If you have idea about cost function, it is similar to it. Here we use Binary Cross Entropy as a loss function. Our aim is to make this loss function zero. To do that we want to make two terms zero(1st term = log(D(x(i))) and 2nd term is log(1 - D(G(z(i)))) )

Now let us consider 1st term,

   To make 1st term zero we want to make D(x(i)) = 1. This means that the discriminator want to detect the training examples as real images. Most probably it is equal to 1. there is no problem with the 1st term.

Now let us consider 2nd term,

   To make 2nd term zero we want to make  D(G(z(i))) = 1. This means that when we pass the fake images generated from the Generator as input to the Discriminator, then Discriminator has to detect that fake images as real images. But how is it possible??

Before knowing the answer, we want to conclude that our loss value is the sum of loss values of two terms i.e., ''how bad the discriminator classifies the real images as real(1st term) and how bad the discriminator classifiers the fake images as real(2nd term).''

Ok, now let's come back to our question, How is it possible to make our 2nd term zero? Here comes the optimizers into the picture. Here we use Adam optimizer to reduce the loss function through back propagation.  Finally, we set up two separate optimizers, one for Generator and one for Discriminator. As specified in the DCGAN paper, both are Adam optimizers with learning rate 0.0002 and Beta1 = 0.5. 

By this way, we reduce our loss, fool the discriminator and generate the fake faces of persons.

## Results
![](Images/results_group.jfif)

![](Images/results_single.png)

### References:
[1] https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html#

[2] https://arxiv.org/pdf/1511.06434.pdf
