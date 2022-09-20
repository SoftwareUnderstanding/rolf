# Starduster
[![Documentation Status](https://readthedocs.org/projects/starduster/badge/?version=latest)](https://starduster.readthedocs.io/en/latest/?badge=latest) [![Build Status](https://app.travis-ci.com/yqiuu/starduster.svg?branch=main)](https://app.travis-ci.com/yqiuu/starduster) [![codecov](https://codecov.io/gh/yqiuu/starduster/branch/main/graph/badge.svg?token=3EX1U22UYW)](https://codecov.io/gh/yqiuu/starduster) [![arXiv](https://img.shields.io/badge/arXiv-2112.14434-blue)](https://arxiv.org/abs/2112.14434)

## Introduction
Starduster is a deep learning model to emulate dust radiative transfer simulations, which significantly accelerates the computation of dust attenuation and emission. Starduster contains two specific generative models, which explicitly take into accout the features of the dust attenuation curves and dust emission spectra. Both generative models should be trained by a set of characteristic outputs of a radiative transfer simulation. The obtained neural networks can produce realistic galaxy spectral energy distributions that satisfy the energy balance condition of dust attenuation and emission. Applications of Starduster include SED-fitting and SED-modelling from semi-analytic models. The code is written in PyTorch. Accordingly, users can take advantage of GPU parallelisation and automatic differentiation implemented by PyTorch throughout the applications.
## Notice
The code is still under heavy development, and therefore the API can be changed in the future. Please raise a issue if you find any bug.
## Installation
The code uses PyTorch. Please go to the [website](https://pytorch.org/) to find an appropriate version to install. After that, clone the repository and install the package by running ``pip install .`` in the repository directory.

