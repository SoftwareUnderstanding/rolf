# epsnoise

## Overview
**epsnoise** simulates pixel noise in weak-lensing ellipticity and shear measurements. It depends mainly on ```numpy```, with some ingredients of ```scipy``` (listed below). Simply place the python file, where the interpreter can find it, e.g. in your ```$PYTHONPATH```.

## Sample generation

**epsnoise** can efficiently create an intrinsic ellipticity distribution, shear it, and add noise, thereby mimicking a "perfect" measurement that is not affected by shape-measurement biases.

A typical workflow with 10,000 simulated galaxies looks like this:

```python
from epsnoise import *
eps_s = sampleEllipticity(10000)
gamma = 0.02 + 0.05j
eps = addShear(eps_s, gamma)
nu = 35
chi_ = addNoise(eps, nu, False)
eps_ = chi2eps(chi_)
```
In this example, ```gamma``` and ```nu``` could also be of type ```numpy.array``` (with the same shape as ```eps```), then shear and noise would be applied to each galaxy individually.

## Marsaglia distribution

For theoretical studies, we provide the Marsaglia distribution, which describes the ratio of normal variables in the general case of non-zero mean and correlation. This requires ```erf``` from ```scipy.special```. We also added a convenience method that evaluates the Marsaglia distribution for the ratio of moments of a Gaussian-shaped brightness distribution, which gives a very good approximation of the measured ellipticity distribution also for galaxies with different radial profiles.

To get the Marsaglia distribution for an image with significance &nu; = 15, showing an object with ellipticity &epsilon; = 0.6, in some range around its true value (in &chi;-space), do the following:
```python
eps = 0.6
nu = 15
chi = eps2chi(eps)
t = arange(chi-0.2, chi+0.2, 0.001)
p_chi = marsaglia_eps(t, 0.6, nu)
```

## Shear estimators

We provide four shear estimators, two based on the &epsilon; ellipticity measure, two on &chi;. While three of them are essentially plain averages, we introduce a new estimator, which requires a functional minimization (```scipy.optimize.fmin```).

The typical mean-of-epsilon estimator, limited to within the unit circle, can be evaluated like this:

```python
eps_s = sampleEllipticity(10000)
gamma = 0.02 + 0.05j
eps = addShear(eps_s, gamma)
nu = 15
eps_ = addNoise(eps, nu)
gamma_est = epsilon_mean(eps_, 0.999)
```

## Reference

More background of this code is presented in the paper 
*Means of confusion: how pixel noise affects shear estimates for weak gravitational lensing* by [Peter Melchior and Massimo Viola, MNRAS, 2012, 424, 2757](http://adsabs.harvard.edu/abs/2012MNRAS.424.2757M)

The theory of the Marsaglia distribution is covered in section 2, the shear estimators in section 4, and the sampling procedure in Appendix B.

If you plan to use this code in a paper, please don't forget to cite us.
