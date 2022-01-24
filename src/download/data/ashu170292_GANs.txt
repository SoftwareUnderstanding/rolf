# GANs and DCGANs
Implementation of GAN and DCGAN paper: 

Through this work I want to explore my understanding of GANs and how they can be used in
widespread applications such as image editing, fake image detection, security, and surveillance,
etc.

MNIST and CelebA datasets were used for this project.

The MNIST database contains 60,000 training images and 10,000 testing images. Half of the
training set and half of the test set were taken from NIST’s training dataset, while the other half
of the training set and the other half of the test set were taken from NIST’s testing dataset

CelebA dataset contains 202,599 face images of various celebrities with diverse facial features and poses.

TRICKS APPLIED TO IMPROVE GAN PERFORMANCE:
some tricks to improve were used to improve GAN performance. These tricks are specifically meant to
improve convergence of our GAN model. 

LABEL SMOOTHING:
One sided label smoothing, i.e., replacing label 1 by 0.9, prevents the neural network from becoming vulnerable to adversarial examples. While this methods is useful in MNIST, this didn’t
work so well with DCGAN.

BATCH NORM, LEAKY RELU AND AVERAGE POOLING:
ReLU activation is replaced by LeakyReLU activation. MaxPooling by Average Pooling layer. While Batch norm helps in faster training of
the GAN by getting rid of the problem of the internal covariate shift. LeakyreLU helps in getting rid of the issue of vanishing gradients. 
Average Pooling yields smoother images as compared to max pooling.

LEARNING RATE:
The generator becomes unstable towards the later epochs.
This means that the loss function of the generator goes away from the optimal because of high
learning rates,thus decrease the learning rate to 2 X 10−5.

RELU IN GENERATOR AND LEAKY RELU IN DISCRIMINATOR:
This was completely based on experiments(no theoritical backing)

CHALLENGES:
The biggest challenge and roadblock in training a GAN is to find stability. A lot of times it may
happen that discriminator is trained way better that the generator and thus the generator finds it
hard to generate fake images. If the discriminator is too weak this would cause all images to be
classified as fake, which will be of no use. RAM issue was faced because of which could not load the (64 X 64) sized image.


REFERENCES\
https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf \
https://arxiv.org/abs/1511.06434
