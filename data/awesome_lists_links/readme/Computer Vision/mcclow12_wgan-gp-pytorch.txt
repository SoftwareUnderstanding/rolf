# wgan-gp-pytorch

This repository contains a PyTorch implementation of the Wasserstein GAN with gradient penalty. 

WGAN works to minimize the Wasserstein-1 distance between the generated data distribution and the real data distribution. This technique offers more stability than the original GAN. 

WGAN-GP improves upon WGAN by using a gradient penalty heuristic rather than weight clipping to encourage the discriminator to be locally 1-Lipschitz near the data manifold.

For more details, see the original paper https://arxiv.org/pdf/1704.00028.pdf.

Some direction was taken from these repositories: https://github.com/arturml/pytorch-wgan-gp, https://github.com/EmilienDupont/wgan-gp. In particular, the Generator and Discriminator architectures were taken from the first repository.

## Usage

To run the code, adjust the hyperparameters in main.py and run

```
python3 main.py
```

This will train the GAN on the MNIST dataset. The necessary dependencies are contained in requirements.txt.

## Generated Images 

Here are some samples from the distribution of the generator:

![alt text](./generated_images/epoch_180.png)

## Plots

These are the plots for the generator loss, discriminator loss, and gradient penalty. These agree with the plots given in this repository: https://github.com/arturml/pytorch-wgan-gp. 

![alt text](./plots/losses.png)
