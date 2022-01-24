
### Wasserstein GAN - Gradient Penalty (WGAN-GP) for MNIST

Based on this Paper: https://arxiv.org/pdf/1704.00028.pdf

#### Brief Introduction

The vanilla GAN (https://arxiv.org/abs/1406.2661) tries to find the Nash Equilibrium between Generator and Discriminator, and it minimizes the Jessen - Shannon Divergence at the optimal point. It is the generative model without the likelihood. However, there were some issues - GAN is very hard to train, and it is precarious. There were many proposed solutions to these problems, as mentioned earlier.

One of the breakthroughs was WGAN paper (https://arxiv.org/abs/1701.07875). Rather than finding the equilibrium between two neural networks, WGAN paper tries to minimize the 1-Wasserstein Distance(WD) between two networks. Intuitively, WD is the cost function of moving one distribution to the another. As the neural network is a powerful function approximator, WGAN finds the optimal transport from the sample to the real distribution.

However, the functions we derived from the WGAN need to meet the 1-Lipschitz condition. WGAN-GP came up with one solution to impose the gradient penalty(GP) as the gradient we obtained from the point between the real data and the samples deviates from 1. This approach works quite well.


#### Implementation

I've implemented WGAN-GP for MNIST data set using PyTorch 1.3.1. I assume that GPUs are available for this implementation and it supports multiple GPUs. You can test by changing the hyperparameters. Sample images are saved for every epoch, and model parameters and losses are recorded periodically.

#### Running

```
python train.py
```

#### Hyperparameters (defaults)

lr : 1e-4 <br/>
wd : True # linearly interpolated between 1e-4 and 0 during the training <br/>
num_epochs : 201 <br/>
latent_dim : 118, # latent dimension for Generator <br/>
ratio : 5, # up to 40 epochs, # Critic is trained 5 times, while Generator is trained once. <br/>
batch : 200, # You may reduce the batch size if there is memory error. <br/>
cp : 0. # Checkpoint, if you train from the certain epoch, you may change this to that epoch.

You can try different hyperparameters by

```
python train.py --lr 1e-2
```

#### Sample images

<1 epoch> <br/>
![1epoch](https://github.com/wayne1123/mnist_wgan_gp/blob/master/imgs/samples1.png) <br/>
<10 epochs> <br/>
![10epochs](https://github.com/wayne1123/mnist_wgan_gp/blob/master/imgs/samples10.png) <br/>
<100 epochs> <br/>
![100epochs](https://github.com/wayne1123/mnist_wgan_gp/blob/master/imgs/samples100.png) <br/>
<200 epochs> <br/>
![200epochs](https://github.com/wayne1123/mnist_wgan_gp/blob/master/imgs/samples200.png)
