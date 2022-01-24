# Introduction
A simple implementation of Generative Adversarial Networks (https://arxiv.org/pdf/1406.2661.pdf) in tensorflow 2 (alpha) on simple 2d point datasets as well as MNIST
Goal is just to see the basics of GAN training as well as tensorflow 2.

Currently, only network models are mulilayer perceptrons (no CNNs).
![Alt text](readme_images/sin.gif?raw=true "Sin data")
![Alt text](readme_images/mnist.gif?raw=true "MNIST data")

# Organization/Usage
*gan.py* contains the main file for GAN related functions. *dataset.py* contains the various datasets to test with. *main.py* is what you would run.

Example main.py run commands:
 - python main.py --data_type sin --disc_iter 2 --gen_iter 1 --batch_size 256 --disc_model 64 32 16 --gen_model 64 32 16 --noise_dim 8
 - python main.py --data_type mnist --disc_iter 2 --gen_iter 1 --batch_size 32 --disc_model 1024 512 256 --gen_model 256 512 1024 --noise_dim 100


# Discussion
GANs are super sensitive to hyper parameter changes and sometimes initial weights. The image generated is cherry picked and you may have to run a couple times to get similar results.

We see issues like mode collapse in both the sinusoid dataset as well as the MNIST dataset. Often, only part of the sinusoid is captured by the generator.

You can see the sensitivity by modifying the example run parameters.


# Dependencies
python3
numpy
tensorflow (2.0.0-alpha0)
matplotlib
argparse

# Todo
 - debug wasserstein loss (https://arxiv.org/pdf/1701.07875.pdf, https://arxiv.org/pdf/1704.00028.pdf)
 - add cnn
