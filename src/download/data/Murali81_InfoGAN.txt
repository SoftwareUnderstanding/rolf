# InfoGAN
A demo script explaining [InfoGAN](https://arxiv.org/pdf/1606.03657.pdf) on MNIST Dataset
![alt_text](https://github.com/Murali81/InfoGAN/blob/master/movie.gif)
Regarding the latent concepts, change the C vector to explore various hidden concepts. 

For a better understanding of InfoGANs, it's better to have grip on GANs, CGANs (Conditional GANs).

### GAN = Generative Adversarial Network ,
has two neural networks , one called as generator and other is a discriminator. The task of generator is to mimic the probability distribution of given dataset. At a high level, a generative model means you have mapped the probability distribution of the data itself. In the case of images, that means you have a probability for every possible combination of pixel values. This also means you can generate new data points by sampling from this distribution ( by choosing combinations with large probability). In Computer vision, this means that we can generate new images entirely from no prior data.

The way it works is similar to a thief and police story. Imagine that a thief always wants to generate fake notes (mimic actual images distribution / mimic actual images (pixel combinations) ) and fool the police to get away with it. Police, on the other hand, wants to determine ways to detect fake notes (To detect a sample that comes from generated probability distribution). It is like a constant mutual development process.

Our stable state is having an equally trained Discriminator (Police to catch fake notes) and Generator (Skilled criminal to mimic currency).

How do we do that ?

In every iteration, 

1. Generator takes a random noise data (vector) and outputs an image (which intially looks like noise). Now you have a bunch of noisy images and true images. You pass this bunch of Noisy images (Label False) and True images (Labelled True) through Discriminator (a neural network). You could see that this turned out to be a simple supervised task of classification. So we train our Discriminator keeping generator untrained.

2. We pass random noise again through Generator to generate fake images. We pass this fake images labelled as True ,(To fool the discriminator) through the discriminator. When passed through Discriminator (Police), we will know the places where we fail to fool the police. Notice the differences and we(Criminal/Generator) work on them. Note that we work on Police's current state of mind, which  means we keep discriminator untrianed during this process.

know the places where we fail to fool the police == Compute loss

Notice the differences and work on them  == Compute gradients and update weights of generator

There's an interesting hypothesis given in DCGAN(Deep Convolutional GAN) paper that there are structures in the random noise vectors which enforce meaningful and consistent impact on the generator. Example given below, is from the DCGAN paper, you could see that when input noise vectors of men are manipulated, images of women with glasses are generated from generator when resulting (manipulated) noise vector is fed. This is impressive and you could also notice that this is similar to arithmetic involved in word embeddings' famous example of King - Man + Woman = Queen

![dc-gan](https://github.com/Murali81/InfoGAN/blob/master/dc_gan.png)

### Conditional GANs :
If you've gone through the above description of GANs, you might have understood that generator generates samples from random noise(Entangled Representation). Wouldn't it be nice if we input a known vector (Disentangled representation) instead of random noise ? Let's say I want to generate handwritten images of a given number. This (Label ===> model ==> image) is the reverse of image classification (image ===> model ==> Label). We are passing in conditional information to the generator for producing images. On the other hand, instead of making the discriminator just classify the images real/fake we pass the label along with the image. Now discriminating, it is classifying whether the label given to the input image is true/not.


## InfoGAN = This is similar to Conditional GANs, but we don't want to specify the information. Let's make neural networks do that for us. But WHY ?

Because when passing real world data like faces,images of buildings, there are a lot of hidden concepts a.k.a Latent concepts. Neural Networks are capable of capturing this latent concepts well because of their non-linear activation functions.

Sounds good. How do we do that ?

For the generator we pass in Z (Random Noise) along with C (categorical distribution). Note that C's distribution may vary with the application to Gaussian / whatever. If it is a categorical distribution, you expect each dimension in C to hold some information regarding shape, rotation etc. For example, [0 0 0 1] may represent a latent concept like rotation. We start with a uniformly distributed C [0.2 0.2 0.2 0.2 0.2].


And with the training process, we expect neural network architecture to update C.

There's only one change compared to GANs & CGANs i.e an additional neural network Q_C to monitor C.

Process is similar,
1.Train discriminator (As usual)
2.Train Generator (As usual)
3.Generate samples from given random noise and initial C. These generated samples are given to Q_C, which in turn learns to embed latent concepts.

# References
[1] Xi Chen, Yan Duan, Rein Houthooft, John Schulman, Ilya Sutskever, Pieter Abbeel, "InfoGAN: Interpretable Representation Learning by
Information Maximizing Generative Adversarial Nets", June 2016 (https://arxiv.org/pdf/1606.03657.pdf)

[2] Alec Radford, Luke Metz and Soumith Chintala, "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks" Jan 2016 (https://arxiv.org/pdf/1511.06434)

[3] Mehdi Mirza, Simon Osindero, "Conditional Generative Adversarial Nets", Nov 2014 (https://arxiv.org/pdf/1411.1784)

[4] Agustinus Kristiadi on InfoGAN (https://wiseodd.github.io/techblog/2017/01/29/infogan/)
