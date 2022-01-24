# GAN (https://mnist-generator.herokuapp.com/)

## Before we begin, check out our [Webapp !](https://mnist-generator.herokuapp.com/)

keras implementation of Generative Adversarial Network for MNIST Digit Generation

<!-- # [Click on this link to Try Now !](https://pong-tfjs.herokuapp.com/ "Pong AI webapp using tfjs") -->

![MNIST GAN Animation](RESULTS.gif?raw=true "MNIST GAN Animation")

# [What are GANs](https://en.wikipedia.org/wiki/Generative_adversarial_network) 

A generative adversarial network (**GAN**) is a class of machine learning systems invented by Ian Goodfellow in 2014.
Two neural networks contest with each other in a game (in the sense of game theory, often but not always in the form of a zero-sum game).
Given a training set, this technique learns to generate new data with the same statistics as the training set.

**For example** <br/>
A GAN trained on photographs can generate new photographs that look at least superficially authentic to human observers, having many realistic characteristics.


**GAN Architecture** <br/>

![GAN_architecture](GAN_architecture.png?raw=true "GAN_architecture")

One neural network, called the generator, generates new data instances, while the other, the discriminator, evaluates them for authenticity; i.e. the discriminator decides whether each instance of data that it reviews belongs to the actual training dataset or not.

Let’s say we’re trying to do something more banal than mimic the Mona Lisa. We’re going to generate hand-written numerals like those found in the MNIST dataset, which is taken from the real world. The goal of the discriminator, when shown an instance from the true MNIST dataset, is to recognize those that are authentic.

Meanwhile, the generator is creating new, synthetic images that it passes to the discriminator. It does so in the hopes that they, too, will be deemed authentic, even though they are fake. The goal of the generator is to generate passable hand-written digits: to lie without being caught. The goal of the discriminator is to identify images coming from the generator as fake.

**Read the original Paper** <br/>
https://arxiv.org/abs/1406.2661 [by -Ian Goodfellow] [[Code](https://github.com/goodfeli/adversarial)]

**Our Implementation** <br/>
We have implemented the above paper using Keras deep learning library

![keras_banner](keras_banner.png?raw=true "keras_banner")

## How it works

1. The generator takes in random numbers and returns an image.
1. This generated image is fed into the discriminator alongside a stream of images taken from the actual, ground-truth dataset.
1. The discriminator takes in both real and fake images and returns probabilities, a number between 0 and 1, with 1 representing a prediction of authenticity and 0 representing fake.


So you have a double feedback loop:

1. The discriminator is in a feedback loop with the ground truth of the images, which we know.
1. The generator is in a feedback loop with the discriminator

## Code


Check out our code at
[gan.py](https://github.com/FnSK4R17s/GAN/blob/master/gan.py) .

check out our webapp at
 [Webapp !](https://mnist-generator.herokuapp.com/) .

## Gallery
Some popular applications of GANs.<br>
1. Colouring Manga 
![Colouring Anime Using GANs](animegan.png?raw=true "Colouring Anime Using GANs")
1. Style Transfer
![Style Transfer](gan_vogh_example2.png?raw=true "Style Transfer")
1. Superresolution using GANs
![Superresolution using GANs](superserolution.png?raw=true "Superresolution using GANs")
