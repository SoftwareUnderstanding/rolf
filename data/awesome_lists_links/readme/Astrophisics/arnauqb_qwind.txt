# Warning: This is an old version of the code no longer supported. Please go [here](https://github.com/arnauqb/Qwind.jl) for the current version.

[![Build Status](https://travis-ci.org/arnauqb/qwind.svg?branch=master)](https://travis-ci.org/arnauqb/qwind)
[![codecov](https://codecov.io/gh/arnauqb/qwind/branch/master/graph/badge.svg)](https://codecov.io/gh/arnauqb/qwind)

Qwind: A non-hydrodynamical model of AGN line-driven winds.
===========================================================


Qwind is a code that aims to simulate the launching and acceleration phase of line-driven winds in the context of AGN accretion discs. To do that, we model the wind as a set of streamlines originating on the surface of the AGN accretion disc, and we evolve them following their equation of motion, given by the balance between radiative and gravitational force.

Code summary
============

Please refer to https://arxiv.org/abs/2001.04720 for a detailed physical explanation of the model. The code is divided into three main classes: <em>wind</em>, <em>radiation</em>, and <em>streamline</em>. The <em>wind</em> class is the main class and the one that is used to initialise the model. It contains the global information of the accretion disc such as accretion rate, density profile, etc. The radiation processes are handled in the <em>radiation</em> class. There are multiple implementations of the radiation class available, for different treatments of the radiation field. A model can be initialised with a particular radiation model by changing the ``radiation_mode`` argument when initialising the wind class. Lastly, the <em>streamline</em> class represents a single streamline, and contains the Euler iterator that solves its equation of motion. It also stores all the physical information of the streamline  at every point it travels, so that it is easy to analyse and debug.

Getting started
===============

Prerequisites
-------------

See the requirements.txt file. Also, due to a current error on the Assimulo package in PyPI, it is required to install it through conda.

```
conda install -c conda-forge assimulo
```

You also need to have installed the GSL library.

Installing
----------

Clone the repo

```
git clone https://github.com/arnauqb/qwind
```

change directory and install with pip,

```
cd qwind
pip install -e .
make
```

Quickstart
----------

See the notebook `quickstart.ipynb` for an example on how to run the code.

Running the tests
=================

The tests can be easily run by installing pytest and running

```
cd test
pytest
```

Citing
======

Please cite the original paper if you use the code. ADS reference: https://ui.adsabs.harvard.edu/abs/2020MNRAS.495..402Q/abstract

License
=======

The project is licensed under the GPL3 license. See LICENSE.md file for details.

