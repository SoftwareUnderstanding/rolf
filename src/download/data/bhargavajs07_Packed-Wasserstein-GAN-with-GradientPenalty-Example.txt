# Packed_WGAN_GP_Example
This repo has the code for implementing Packed Wasserstein GAN with Gradient Penalty.
This code implements the collection of ideas proposed in the following papers
1. WGAN: https://arxiv.org/pdf/1701.07875.pdf
2. WGAN_GP: https://arxiv.org/pdf/1704.00028.pdf
3. Pac_GAN: https://arxiv.org/pdf/1712.04086.pdf

In this example, a model is trained to generate samples from a 1-D Mixture of Gaussians supported on a single line in R2 (using MoG instead of Uniform distribution in Example-1 of paper-1).
The intent of this exercise to see how these models help in avoiding the mode collapse problem when training to generate  samples from a multi-modal distribution supported on low-dimensional manifold.
It is observed in this example that the packing of input samples input to the critic function aids well in avoiding the mode-collapse problem.

