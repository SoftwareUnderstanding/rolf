<a href="http://ascl.net/1711.006"><img src="https://img.shields.io/badge/ascl-1711.006-blue.svg?colorB=262255" alt="ascl:1711.006" /></a>
<a href="https://cran.r-project.org/package=rgw"><img src="https://img.shields.io/cran/v/rgw.svg" alt="CRAN" /></a>
<a href="https://raw.githubusercontent.com/abmantz/rgw/master/LICENSE"><img src="https://img.shields.io/cran/l/rgw.svg" alt="MIT License" /></a>

# rgw

This package implements in [R](https://www.r-project.org/)  the affine-invariant sampling method of [Goodman & Weare (2010)](http://dx.doi.org/10.2140/camcos.2010.5.65). This is a way of producing Monte-Carlo samples from a target distribution, which can be used for statistical inference.

This R implementation is based on the very clear description given by [Foreman-Mackey et al. (2012)](https://arxiv.org/abs/1202.3665), who provide an implementation [in python](http://dan.iel.fm/emcee).

## Installation

### From CRAN

In R, run ```install.packages("rgw")```. Note that the version hosted on CRAN may lag behind this one (see [VERSION.md](VERSION.md)).

### Manually (Linux/Unix/Mac)

1. Clone this repository.
2. In a terminal, navigate to the ```<repository base>/R/```.
3. Run ```R CMD install rgw```. Alternatively, in an R session, run ```install.packages("rgw", repos=NULL)```.

## Use

Here's the simple example that appears in the documentation:

```R
# In this example, we'll sample from a simple 2D Gaussian.

# Define the log-posterior function
lnP = function(x) sum( dnorm(x, c(0,1), c(pi, exp(0.5)), log=TRUE) )

# Initialize an ensemble of 100 walkers. We'll take 100 steps, saving the ensemble after each.
nwalk = 100
post = array(NA, dim=c(2, nwalk, 101))
post[1,,1] = rnorm(nwalk, 0, 0.1)
post[2,,1] = rnorm(nwalk, 1, 0.1)

# Run
post = GoodmanWeare.rem(post, lnP)

# Plot the final ensemble
plot(post[1,,101], post[2,,101])
# Look at the trace of each parameter for one of the walkers.
plot(post[1,1,])
plot(post[2,1,])
# Go on to get confidence intervals, make niftier plots, etc.
```

## Help

Open an [issue](https://github.com/abmantz/rgw/issues).
