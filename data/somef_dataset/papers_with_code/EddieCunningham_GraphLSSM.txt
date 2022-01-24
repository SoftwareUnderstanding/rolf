# Graphical Latent State Space Models

This is a Python library that extends traditional generative time series models, such as hidden markov models, linear dynamical systems and their extensions, to graphical models.

With this framework, a user can do Bayesian inference over graphical structures.  One use case is doing inference over a [pedigree chart](https://en.wikipedia.org/wiki/Pedigree_chart), where phenotypes (observations) are emitted based on each person's genotype (latent state), and the genotypes of individuals are linked through their ancestor tree.

Here's what is currently implemented:
  - Discrete Latent States
  - Inference over latent states on DAGs
  - Gibbs Sampling
  - Expectation Maximization
  - Coordinate Ascent Variational Inference
  - Stochastic Variational Inference
  - [SVAE](https://arxiv.org/abs/1603.06277) for non conjugate observations using [Autograd](https://github.com/HIPS/autograd) and the [Gumbel Softmax trick](https://arxiv.org/pdf/1611.01144.pdf)
  - Directly optimize marginal probability P(Y;ϴ) for model with non-conjugate observations
  - Visualization with Graphviz
  
<img src="examples/hmm_em.gif">
