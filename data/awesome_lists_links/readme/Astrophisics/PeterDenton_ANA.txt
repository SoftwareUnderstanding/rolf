ANA: Astrophysical Neutrino Anisotropy
=
| master | version | DOI |
|:------:|:-------:|:---:|
|[![Build Status](https://travis-ci.org/PeterDenton/ANA.svg?branch=master)](https://travis-ci.org/PeterDenton/ANA)|[![GitHub Version](https://badge.fury.io/gh/PeterDenton%2FANA.svg)](http://badge.fury.io/gh/PeterDenton%2FANA)|[![DOI](https://zenodo.org/badge/84968421.svg)](https://zenodo.org/badge/latestdoi/84968421)|

## Overview
This code has resulted in the publication [arXiv:1703.09721](https://arxiv.org/abs/1703.09721). This code calculates the likelihood function for a model comprised of two components to the astrophysical neutrino flux detected by IceCube. The first component is extragalactic. Since point sources have not been found and there is increasing evidence that one source catalog cannot describe the entire data set, we model the extragalactic flux as isotropic. The second component is galactic. A variety of catalogs of interest exist here as well. We take the galactic contribution to be proportional to the matter density of the universe.

The likelihood function has one free parameter f<sub>gal</sub> that is the fraction of the astrophysical flux that is galactic. The code finds the best fit value of f<sub>gal</sub> and scans over 0&lt;f<sub>gal</sub>&lt;1.

## Installing
See the [dependency guide](DEPENDENCY.md) and the [installation guide](INSTALL.md).

## About the code
The IceCube events are read in and managed in [*src/ICEvent.cpp*](src/ICEvent.cpp). The likelihood function is described in [*src/Backgrounds.cpp*](src/Backgrounds.cpp) and [*src/Likelihood.cpp*](src/Likelihood.cpp). The von Mises-Fisher distribution is handled in [*vMF.cpp*](src/vMF.cpp). A Markov Chain Monte Carlo code to generate points distributed in the galactic plane is in [*src/MWDisks.cpp*](src/MWDisks.cpp). Finally, [*src/Figures.cpp*](src/Figures.cpp) generates data files that can then be plotted by the scripts in the [*py*](py) directory and [*src/main.cpp*](src/main.cpp) indicates what should be run.

## Further details
1. Calculating the integral in equation 4.6 in the paper is the main time consuming portion of the code, other than generating the sky map and the galactic plane visualization. With this in mind, the code writes the calculation to file, *data/L_gals.txt* with `calc_L_gals()` and the result is read in to a global variable with `read_in_gals()`.

1. The high energy cut on the galactic component mentioned in the paper can be turned on with the bool flag in `calc_L_gals()`.

1. To turn off the progress bar on the slower functions, uncomment the `Progress_Bar_visible = false;` line in [*src/main.cpp*](src/main.cpp).

1. To turn on a broken power law, change `Phi_astro` in [*src/Backgrounds.cpp*](src/Backgrounds.cpp) and modify `Phi_astro2` as necessary.

## Support
If you have questions or encounter any problems when running *ANA*, please use github's [issue tracker](https://github.com/PeterDenton/ANA/issues).

This code is free to use, copy, distribute, and modify.
If you use this code or any modification of this code, we request that you reference both this code [DOI:10.5281/zenodo.438675](https://zenodo.org/record/438675) and the paper [arXiv:1703.09721](https://arxiv.org/abs/1703.09721).
