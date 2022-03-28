# DLA_interview

## Introduction
The objective of this project is to learn more about conditional generative models. Having worked with GANs, it seems beneficial to study more about adding additional descriptive information with the input image to produce models that are able to distinctly represent specific subjects in the generated data. It seems to be a part of how users can select specific features or labels for the model to generate. As an early step of looking at this and taking into account the limitations of resources and time, this project will be experimenting with the vanilla variational autoencoder and a conditional variational autoencoder.


## Installation changed:
Install pytorch from here:
https://devtalk.nvidia.com/default/topic/1049071/jetson-nano/pytorch-for-jetson-nano/

For the Jetson TX2, Pytorch cannot be installed using the method described on the above Github page. An alternate version for the GPU must be downloaded from the NVIDIA site as indicated by the link above. 

## Process
The original Variational Autoencoder paper and code implemeted in pytorch and the accompanying paper which is initially applied to the MNIST. Since MNIST is a dataset that has been implemented many times and the different classes can be identified with only a few pixels, the variational autoencoder will also be applied to the FashionMNIST data and KMNIST data to have a better understanding of performance.

Original VAE paper and accompanying code example
https://arxiv.org/pdf/1312.6114.pdf

https://github.com/pytorch/examples/tree/master/vae

The paper on the conditional variational autoencoder and it's loss function is as follows
https://pdfs.semanticscholar.org/3f25/e17eb717e5894e0404ea634451332f85d287.pdf

To implement a conditional variational autoencoder, the original varaiational autoencoder is modified several ways:

* The input size of the encoder neural network is increased by the number of labels. The digit label is one hot encoded and concatenated to the initial input size of 28 * 28 = 784 so the input of the encoder network is 28 * 28 + 10 - 794.
* The input size of the decoder neural network is increased by the number of labels. For the original MNIST network a latent variable size of 2 was chosen, so the input to the decoder network is now 2 + 10 = 12.
* All additional layers and nodes of the networks remain the same

The CVAE network is further modified to have the label data concatenated to the inputs and the reparametrized latent variables. The loss function is still calculated over the same features and does not change with label data.

All experiments were run for 200 epochs with a learning rate of 0.001. The hidden layer on the encoder network and the decoder network have 500 nodes as described in the variational autoencoder paper for experiments done on the MNIST data. Changing the number of nodes did not seem to make any discernible difference on the images as seen by a person, so there was no need to adjust. The latent variable size for the MNIST data set was set to 2. Since the KMNIST and FashionMNIST datasets had more detail, the latent variable size was adjusted to 10.

A few experiments were run on the CIFAR10 data set as well, but due to poor performance of variational autoencoders on these images, those results were not pursued in this project. 


## Results
The conditional variational autoencoder always prints out the correct digit or article of clothing for the FashionMNIST data. This is likely becasue the label data is encoded in the input of the encoder. When the latent space is generated, it enocdes each digit as a separate Gaussian function where Z\~N(0,I). In the vanilla variational autoencoder, all digits are encoded to the same Z\~N(0,I), where different digits are clustered. This makes points that line near the boundaries of different digits less discernible. When checking even later samples of reconstructed test points, examples of digits that differ in value can be seen.

In all figures of reconstructed images, the first row are original images taken from the MNIST datasets. The second row are those reconstructed by the conditional Variational Autoencoder and the last row are reconstructions done by the vanilla VAE.

![MNIST reconstructions](resultsMNIST/reconstruction_199.png)

In the above figure the first and last digits is clearly a 4 and the conditinoal VAE is able to reconstruct a 4, the images reconstructed by the vanilla VAE are closer to 9's. Because there is an additional set of one hot encoded labels in the conditional vae, the Gaussian distibuted latent variable space does not overlap, whereas with the vanilla VAE the latent space points representing 4's and 9's have some overlap, so the model may generate incorrect digits.

The conditional vraiational autoencoder also allows for selecting which digit will be represented by the generated data.

![MNIST VAE](resultsMNIST/VAE_sample_199.png)
![MNIST CVAE](resultsMNIST/CVAE_sample_199.png)

In addition to selecting the label for the data represented, the loss function for the CVAE is improved over the VAE. This seems to be due to the additional condition in the estimated generative model. In the calculation of the loss, while the size of the space, it is calculated on is the same for both functions, the CVAE is conditioned on both the input and the label, this additional information decreases the loss for all data sets.

![MNIST](MNIST.png) ![FMNIST](FMNIST.png) ![KMNIST](KMNIST.png)


The following reconstructions were generated for the Fashion MNIST data and the KMNIST data at 199 epochs.

![FMNIST reconstructions](resultsFMNIST/reconstruction_199.png)
![KMNIST reconstructions](resultsKMNIST/reconstruction_199.png)

There is some improvement in the FMNIST data and KMNIST data, this may be improved with changes in the latent variable space. The VAE inherently does not produce sharp images, and the CVAE does not improve this significantly. As seen in the FMNIST images, detailed patterns on the clothing is lost in reconstructions. It's a good idea to look at otehr generative models for these datasets.

The code was also run on the CIFAR10 data, but a few runs stopped due to running out of RAM, and any reconstructions produced were unrecognizable for both the VAE and CVAE.

## Future Work
The variational autoencoder was chosen for this proejct due to resources and ease of implementation. The work done here only added a discrete variables (labels), but some of the images I've worked with contained information that would not be a label (angle). I think it would be interesting to see if this would potentially improve performance if there was some way to add this information to the input in image processing problems.

I initially wanted to try out using Real NVP at https://arxiv.org/pdf/1605.08803.pdf and potentially generating labeled images since it produces sharper images than the variational autoencoder, but consists of a similar structure, where we tranform data to a Gaussian distribution and back. But this technique took a little too long to run. I think it may be interesting to create a conditional real NVP model.


## Demo

For a short version of the code. Open the terminal on the computer and cd into the folder with demo.py and run in python3 with the following command. The demo trains on the MNIST data for 10 epochs and with 6000 training samples and 1000 training samples.

```
cd vae
python3 demo.py
```

or open terminal and run

```
bash demo.sh
```
