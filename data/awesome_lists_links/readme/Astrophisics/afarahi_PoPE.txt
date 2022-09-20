![GitHub](https://img.shields.io/github/license/afarahi/PoPE)
![PyPI](https://img.shields.io/pypi/v/pope)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pope)
<a href="http://ascl.net/2007.006"><img src="https://img.shields.io/badge/ascl-2007.006-blue.svg?colorB=262255" alt="ascl:2007.006" /></a>


<p align="center">
  <img src="logo.png" width="300" title="logo">
</p>


# Introduction

The spatial distribution and internal structure of astronomical systems contain vast amount of information about the
 underlying physics that governs the formation, evolution, and fate of these systems. While astronomical data collected
 by large-, medium-, and small-size surveys are becoming more abundant, precise and accurate modeling is becoming more
 challenging. The scale and complexity of these multi-wavelength surveys are exceeding the capabilities of traditional
 data analysis models, hence the need for novel inference models. 

One of the key challenges of modeling the empirical data is how to account for the measurement errors of varying 
magnitudes. The low signal-to-noise ratio (SNR) regime hinders the ability to infer the spatial structure of a 
population from abundant but noisy measurements, diluting the spatial signals. Typical measurements with SNR below 
the detection limit are often discarded or stacked to boost the signal above the detection limit. Binning and stacking 
can introduce selection bias, information loss, and smearing out the signal component. While stacking amplifies the 
SNR of the population average properties, it suppresses intrinsic scatter of the population under study. 
In practice, performing a statistical inference on large astronomical datasets has become a bottleneck of traditional 
population- and likelihood-based approaches.


To address some of the above challenges, we developed Population Profile Estimator (`PoPE`), a population-based, 
Bayesian inference model to analyze a class of problems that are concerned with the spatial distribution or internal 
spatial structure of a sample of astronomical systems. Our method uses the conditional statistics of spatial 
profile of multiple observables assuming the individual observations are measured with errors of varying magnitude. 
Assuming the conditional statistics of our observables can be described with a multivariate normal distribution, 
the model reduces to the conditional average profile and conditional covariance between all observables. 
The method consists of two steps: (1) reconstructing the average profile using non-parametric regression with Gaussian 
Processes and (2) estimating the the property profiles covariance given a set of independent variable. Our 
population-based method is computationally efficient and capable of inferring average profiles of a population from 
noisy measurements, without stacking and binning nor parameterizing the shape of the average profile. This code is an 
implementation of Population Profile Estimator (`PoPE`) method that performs a regression analysis described in 
Farahi, Nagai & Cheng (2020). 

If you use PoPE or derivates based on it, please cite the following paper ([Farahi et al. 2020](https://arxiv.org/abs/2006.16408)) 
which introduced the tool.

## Dependencies

`numpy`, `scipy`,  `matplotlib`, `pandas`, `sklearn`, `pymc3`, `KLLR`.

## References

[1]. A. Farahi, D. Nagai, Y. Cheng, "PoPE: A population-based approach to model spatial structure of astronomical 
systems", AJ 161 30 (2021). [arXiv link](https://arxiv.org/abs/2006.16408). [Journal link](https://iopscience.iop.org/article/10.3847/1538-3881/abc630/pdf).

## Installation

Run the following to install:
  
    pip install pope

## Quickstart

To start using `PoPE`, simply use `from PoPE import estimate_mean_property_profile` to
access the primary functions and class. The exact requirements for the inputs are
listed in the docstring of the estimate_mean_property_profile() class further below.
An example for using `PoPE` looks like this:
                                                                        
        from PoPE import estimate_mean_property_profile                                       
                                                                          
        # load data and add measurement noise
        # Xs, Ys1, Ys2, Ys1err, Ys2err = load_data()

        # compute the average profile
        mp, gp, model = estimate_mean_property_profile(Xs, Ys1, Ys2, Ys1err, Ys2err,
                                                       Xu_shapes=[15, 7], kernel_scales=[2.0, 2.0])                                 
                                                                          

See examples "./examples/simulated_example.py" and "./examples/TNG_example.py" for more information. To replicate 
plots presented in the paper, you can run `python3 TNG_profile_example.py` and `python3 fake_simulated_example.py` 

## Contact

If you have any questions or want to modify the code for your own purpose, please do not hesitate to 
email arya.farahi@austin.utexas.edu for help.
