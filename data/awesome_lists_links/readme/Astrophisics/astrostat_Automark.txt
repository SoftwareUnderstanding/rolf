# Automark

This package implements a method for modeling photon counts collected form
observation of variable-intensity astronomical sources. It aims to **mark** the
abrupt changes in the corresponding wavelength distribution of the emission
**automatically**. In the underlying methodology, change points are embedded into a
marked Poisson process, where photon wavelengths are regarded as marks and both
the Poisson intensity parameter and the distribution of the marks are allowed
to change. The details is given in Wong et. al. (2016).

## Installation
This package can be installed via function `install_github` in R package `devtools`:

``` r
install.packages("devtools")
devtools::install_github("astrostat/Automark")

```

We plan to upload this package to CRAN in the future. That is, future installation can
also be done by `install.packages("Automark")` within R.

## Example
```r
library(Automark)       # load library
data(dat)               # load example data
out <- spec.tbreak(dat$x1, dat$x2, dat$y, dat$A, dat$delta.t, dat$delta.w, cpus=5)
plotspec(dat$x2, out$best.fit[[1]])
plotspec(dat$x2, out$best.fit[[2]], np=F, col="red")
```


## References
* Raymond K. W. Wong, Vinay L. Kashyap, Thomas C. M. Lee and David A. van Dyk (2016).
Detecting Abrupt Changes in the Spectra of High-energy Astrophysical Sources. The Annals of Applied Statistics, 10(2), 1107-1134. [\[Journal\]](http://dx.doi.org/10.1214/16-AOAS933) [\[arXiv\]](http://arxiv.org/abs/1508.07083)
