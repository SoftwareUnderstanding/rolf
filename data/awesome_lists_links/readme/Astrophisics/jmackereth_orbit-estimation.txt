# orbit-estimation

Code for testing and evaluating the Stäckel approximation method for estimating orbit parameters in galactic potentials, as 
described in [Mackereth & Bovy (2018, in prep)](https://arxiv.org/abs/1802.02592v1). The method itself is implemented in [galpy](https://github.com/jobovy/galpy); see galpy's documentation for instructions on how to use it.

## Introduction

While action-angle variables are now an essential tool in Galactic dynamics, orbital parameters such as eccentricity and maximum vertical excursion are still useful tools for understanding the dynamics and evolution of the Galaxy. The computation of these parameters for galactic potentials has usually required computationally expensive orbit integration. Here, we demonstrate and test a new method for their estimation using the Stäckel approximation. The method relies on the approximation of the Galactic potential as a Stäckel potential, in a prolate confocal coordinate system, under which the vertical and horizontal motions decouple. By solving the Hamilton Jacobi equations at the turning points of the horizontal and vertical motions, it is possible to determine the spatial boundary of the orbit, and hence calculate the desired orbit parameters.

This repo includes all the code to generate the plots and tests in the paper, as well as some extra explorations into the method and its application to observational data. Running the code requires the usual `scipy` ecosystem packages, `astropy`, Jo Bovy's [`gaia_tools`](https://github.com/jobovy/gaia_tools) and a full installation (including the C extensions) of [`galpy`](https://github.com/jobovy/galpy).

## Main Code

[**orbit_helper.py**](https://github.com/jmackereth/orbit-estimation/blob/master/py/orbit_helper.py) - definition of helper functions which are used in the evaluation of the method. Includes functions for quickly initialising a grid of orbits in energy and angular momentum space, and performing an integration vs estimation comparison for a given orbit.

[**mcmillan.py**](https://github.com/jmackereth/orbit-estimation/blob/master/py/mcmillan.py) - defines the use of `galpy`'s Self Consistent Field (SCF) method implementation to generate an approximation of the [McMillan (2017)](http://adsabs.harvard.edu/abs/2017MNRAS.465...76M) Milky Way potential, which is used in the paper as a more complex alternative to `galpy`'s `MWPotential2014`. This file uses the subclasses of `SCFPotential` and `DiskSCFPotential` in [SCF_derivs.py](https://github.com/jmackereth/orbit-estimation/blob/master/py/SCF_derivs.py) which add the numerically computed second derivatives to these potentials.

[**orbit-estimation.ipynb**](https://github.com/jmackereth/orbit-estimation/blob/master/py/orbit-estimation.ipynb) - the main notebook used to generate the plots included in the paper, and perform further exploration of the method. 

[**McMillan_pot_tests.ipynb**](https://github.com/jmackereth/orbit-estimation/blob/master/py/McMillan_pot_tests.ipynb) - tests of the implementation of the McMillan (2017) Milky Way potential, and comparisons with orbit integration and parameter estimation in `MWPotential2014`.

## Extras

[**dierickx_eccentricities.py**](https://github.com/jmackereth/orbit-estimation/blob/master/py/dierickx_eccentricities.py) - code that estimates and plots the eccentricity distribution as measured by [Dierickx et al. (2010)](http://adsabs.harvard.edu/abs/2010ApJ...725L.186D). Use the script [get_dierickx.py](https://github.com/jmackereth/orbit-estimation/blob/master/py/get_dierickx.py) to download the necessary data.




