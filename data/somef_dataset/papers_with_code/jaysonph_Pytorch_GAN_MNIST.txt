# Pytorch_GAN_MNIST
Pytorch implementation of Basic GAN in generating handwritten digits

## Background
Generative Adversarial Net (GAN) is a deep learning architecture framework that consists of 2 main components - Generator and Discriminator. It is an extremely powerful architecture in many content-generation tasks (e.g. image generation, Low-light image enhancement).

GAN was introduced in a paper (https://arxiv.org/pdf/1406.2661.pdf) by Ian Goodfellow and other researchers including Yoshua Bengio in 2014. Facebook's AI research director Yann LeCun once said adversarial training being "the most interesting idea in the last 10 years in ML"

## Training
In this project, I have carried out 300 epochs of training. In the process, I have encountered different problems (e.g. too high learning rate, overfitting of the Discriminator).
At last, the training went well. The losses are shown below:

![GAN_MNIST_losses](https://user-images.githubusercontent.com/40629085/69151138-05569300-0b15-11ea-8618-807bcea46a99.jpeg)

## Results
Below is the evolution of the result in 300 epochs (sampled every 10 epochs)

![GAN_MNIST_300_epochs](https://user-images.githubusercontent.com/40629085/69150921-9f6a0b80-0b14-11ea-9d6e-551d9f4cc9fe.gif)
