velbin
======

Overview
========

This code convolves the radial velocity offsets due to binary orbital motions with a Gaussian to model an observed velocity distribution. This can be used to measure the mean velocity and velocity dispersion from an observed radial velocity distribution, corrected for binary orbital motions. For detailed information on the technique and many tests see Cottaar et al. (2012, A&A, 547, 35) and Cottaar & Henault-Brunet (2014, A&A, 562, 20). Here we only give information necesary to fit the code

This code has been designed to fit single- or multi-epoch data with any arbitrary binary orbital parameter distribution (as long as it can be sampled properly), however it always assumes that the intrinsic velocity distribution (i.e. corrected for binary orbital motions) is a Gaussian.

The code is completely unrelated to the Russian village of the same name

Installation
============
The code can be downloaded from github using git (note that git will have to be installed):
```shell
git pull https://github.com/MichielCottaar/velbin.git
```

The default python installation should work. In short move into the downloaded directory and run
```shell
python setup.py install
```
This will copy velbin to the python library, making it available for importing in python

Getting started
===============
Velbin is limited to three main features:
1. Sample (and edit) a binary orbital parameter distribution. This is a prerequisite for the other two features.
2. Fit an observed radial velocity distribution
3. Create a mock radial velocity distribution.

1. Sampling and editing a binary orbital parameter distribution
===============================================================
After importing velbin, a variety of functions are available to sample an initial orbital parameter distribution (including a period, mass ratio, eccentricity, phase, theta, and inclination distribution, although the orientation (i.e. theta and inclination) distributions are ignored when fitting single-epoch data):
- `velbin.solar`: provides distributions appropriate for solar-type stars
- `velbin.ob_stars`: provides distributions appropriate for OB stars

All of these return an OrbitalParameters object, which is a subclass from the numpy record array. All of the numpy array goodies are available to edit the orbital parameter distribution. In addition to directly editing the array, three helper methods are provided:
- `draw_period`: sample a period distribution
- `draw_mass_ratio`: sample a mass ratio distribution
- `draw_eccentricities`: sample an eccentricity distribution

In addition the method `semi_major` computes the semi-major axes of the sampled binaries. There is no support to set the semi-major axes distribution directly. The method `velocity` computes the radial velocity and (instanteneous) proper motion due to binary orbital motions.

Example: 
Draw an orbital parameter distribution from Raghavan et al. (2010, ApJS, 190, 1), using the a flat mass ratio distribution and a lower limit to the semi-major axis of 1 AU (as might be appropriate for solar-mass red giants):
```python
import velbin
all_binaries = velbin.solar('Raghavan10')
all_binaries.draw_mass_ratio('flat')
selected_binaries = all_binaries[all_binaries.mass_ratio() > 1.]
```

2. Fitting an observed radial velocity distrution
=================================================
The distributions of radial velocity offsets due to binary orbital motions are computed by two OrbitalParameters methods:
- `single_epoch`: Computes the distribution of radial velocity offsets for all binaries as prepartion for fitting a single-epoch radial velocity dataset.
- `multi-epoch`: Identifies the seemingly single star, identifies for every seemingly single star which binaries would not have been detected, and computes their radial velocity offset distribution. Uses a chi-squared test to distinguish RV-variables from seemingly single stars (the p-value threshold can be set and is 1e-4 by default).
Both expect the observed radial velocity distribution, the measurement uncertainties, and the masses of the observed stars to be provided.

Both of these methods take return a BinaryFit object. Given a value for the mean velocity, velocity dispersion, and binary fraction of the cluster this object can be called to compute the log-likelihood to reproduce the provided radial velocity distribution. Maximizing this log-likelihood (using e.g. the routines in scipy.optimize or openopt) provides the best-fit values of the mean velocity, velocity dispersion, and binary fraction. Note that a best-fit binary fraction of 100% generally implies an overestimation of the velocity dispersion (Cottaar & Henault-Brunet, 2013, in preperation).

3. Create a mock radial velocity dataset
========================================
Given an orbital parameter distribution stored in an OribitalParameters object `all_binaries`, a mock single-epoch radial velocity dataset of 10 radial velocities with velocity dispersion `vdisp` and binary fraction `fbin` can be created by
```python
mock_dataset = all_binaries.fake_dataset(5, vdisp, fbin, sigvel=1.)
```
where we have set the measurement uncertainties (i.e. `sigvel`) to one km/s.

A multi-epoch dataset can be simply created by setting the `dates` keyword parameter to an iterable with more than one element. The code
```python
two_epoch_dataset = all_binaries.fake_dataset(5, vdisp, fbin, sigvel=1., dates=(0, 1))
```
will create a dataset with two epochs, which are one year apart.

These datasets can immediately used to provide the fitted radial velocities in the `single_epoch` or `multi_epoch` methods.
