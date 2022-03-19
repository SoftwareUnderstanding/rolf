# ApproximateVI.jl

Approximate variational inference in Julia

This package implements approximate variational inference as presented in  
*Approximate variational inference based on a finite sample of Gaussian latent variables,  
Pattern Analysis and Applications volume 19, pages 475–485, 2015* [[DOI]](https://doi.org/10.1007/s10044-015-0496-9), [[Arxiv]](https://arxiv.org/pdf/1906.04507.pdf).

**Documentation and more functionality will be added to this repository soon**


## What is this package about

This package implements variational inference using the re-parametrisation trick.
The work was independently developed and published [here](https://doi.org/10.1007/s10044-015-0496-9). 
Of course, the method has been widely popularised by the works [Doubly Stochastic Variational Bayes for non-Conjugate Inference](http://proceedings.mlr.press/v32/titsias14.pdf) and [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114).
The method indepedently appeared earlier in [Fixed-Form Variational Posterior Approximation through Stochastic Linear Regression](https://arxiv.org/abs/1206.6679) and later in [A comparison of variational approximations for fast inference in mixed logit models](https://link.springer.com/article/10.1007%2Fs00180-015-0638-y) and very likely in other publications too...


## What does the package do

The package offers function `VI`. This function approximates the posterior parameter distribution
with a Gaussian q(θ) = 𝜨(θ|μ,Σ) by maximising the expected lower bound:

∫ q(θ) log p(x,θ) dθ + ℋ[q]

The above integral is approximated with a Monte carlo average of S samples:

1/S 𝜮ₛ log p(x,θₛ) dθ + ℋ[q]

Using the reparametrisation trick, we re-introduce the variational parameters that we need to optimise:

1/S 𝜮ₛ log p(x,μ + √Σ zₛ) dθ + ℋ[q], where √Σ is a matrix root of Σ, i.e. √Σ*√Σ' = Σ, and zₛ∼𝜨(0,I).

Contrary to other flavours of this method, that repeatedly draw new samples zₛ at each iteration of the optimiser, here a large number of samples zₛ is drawn
at the start and kept fixed throughout the execution of the algorithm (see [paper](https://arxiv.org/pdf/1906.04507.pdf), Algorithm 1).
This avoids the difficulty of working with a noisy gradient and allows the use of optimisers like [LBFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS). However, this comes at the expense of risking overfitting to the samples zₛ that happened to be drawn. A mechanism for monitoring potential overfitting is described in the [paper](https://arxiv.org/pdf/1906.04507.pdf), section 2.3. Because of fixing the sample zₛ, the algorithm doesn't not scale well to high number of parameters and is thus recommended for problems with relatively few parameters, e.g. 2-20 parameters. Future work may address this limitation. A method that partially addresses this limitation has been presented [here](https://arxiv.org/abs/1901.04791). 


## How to use the package

The package is fairly easy to use. The only function of interest to the user is `VI`. At the very minimum, the user needs to provide a function that codes the joint log-likelihood function.

Consider, approximating a target density given by a three-component mixture model:

```
using PyPlot # Necessary for this example

# Define means for three-component Gaussian mixture model
# All components are implicitly equally weighted and have unit covariance
μ = [zeros(2), [2.5; 0.0], [-2.5; 0.0]]

# Define log-likelihood
logp(θ) = log(exp(-0.5*sum((μ[1].-θ).^2)) + exp(-0.5*sum((μ[1].-θ).^2)) + exp(-0.5*sum((μ[3].-θ).^2)))
```

We will now approximate it with a Gaussian density. We need to pass to ```VI``` the log-likelihood function, a starting point for the mean of the approximate Gaussian posterior, as well as the number of fixed samples and the number of iterations we want to optimise the lower bound for:

```
posterior, logevidence = VI(logp, randn(2); S = 100, iterations = 30)
```

This returns two outputs: the first one is the approximating posterior q(θ) of type ```MvNormal``` (see [Distributions.jl](https://github.com/JuliaStats/Distributions.jl)). The second output is the approximate lower bound of type ```Float64```.

Below we plot as contour plot the target unnormalised posterior distribution.
We also plot the approximating posterior q(θ) as a blue ellipse:

![image](docs/images/examplemixturemodel_ellipse.png)


## Further examples
More examples can be found in the [/src/Examples](/src/Examples) folder.


