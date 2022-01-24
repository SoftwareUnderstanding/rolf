# GAN_Using_Pytorch

Generative Adversarial Networks

What is a GAN?

GANs are a framework for teaching a DL model to capture the training data’s distribution so we can generate new data from that same distribution. 
GANs were invented by Ian Goodfellow in 2014 and first described in the paper Generative Adversarial Nets. 
They are made of two distinct models, a generator and a discriminator. The job of the generator is to spawn ‘fake’ images that look like the training images. 
The job of the discriminator is to look at an image and output whether or not it is a real training image or a fake image from the generator.
 During training, the generator is constantly trying to outsmart the discriminator by generating better and better fakes, while the discriminator is working to become a better detective and correctly classify the real and fake images. T
 he equilibrium of this game is when the generator is generating perfect fakes that look as if they came directly from the training data, and the discriminator is left to always guess at 50% confidence that the generator output is real or fake.
 D(G(z)) is the probability (scalar) that the output of the generator G is a real image. 
 As described in Goodfellow’s paper, D and G play a minimax game in which D tries to maximize the probability it correctly classifies reals and fakes (logD(x)), and G tries to minimize the probability that D will predict its outputs are fake (log(1−D(G(z)))). From the paper, the GAN loss function is


Generator:

The generator, GG, is designed to map the latent space vector (zz) to data-space. 
Since our data are images, converting zz to data-space means ultimately creating a RGB image with the same size as the training images (i.e. 3x64x64). 


Discriminator:

The discriminator, DD, is a binary classification network that takes an image as input and outputs a scalar probability that the input image is real (as opposed to fake). 
Here, DD takes a 3x64x64 input image, processes it through a series of Conv2d, BatchNorm2d, and LeakyReLU layers, and outputs the final probability through a Sigmoid activation function. 



Training:

we will do some statistic reporting and at the end of each epoch we will push our fixed_noise batch through the generator to visually track the progress of G’s training. The training statistics reported are:

Loss_D - discriminator loss calculated as the sum of losses for the all real and all fake batches (log(D(x)) + log(1 - D(G(z)))log(D(x))+log(1−D(G(z)))).
Loss_G - generator loss calculated as log(D(G(z)))log(D(G(z)))
D(x) - the average output (across the batch) of the discriminator for the all real batch. This should start close to 1 then theoretically converge to 0.5 when G gets better. Think about why this is.
D(G(z)) - average discriminator outputs for the all fake batch. The first number is before D is updated and the second number is after D is updated. These numbers should start near 0 and converge to 0.5 as G gets better. Think about why this is.


UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS: https://arxiv.org/abs/1511.06434

Dataset: https://drive.google.com/file/d/1VT-8w1rTT2GCE5IE5zFJPMzv7bqca-Ri/view
