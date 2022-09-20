# spotrod
### A semi-analytic model for transits of spotted stars.

```
Copyright 2013, 2014 Bence Béky

This file is part of Spotrod.

Spotrod is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Spotrod is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Spotrod.  If not, see <http://www.gnu.org/licenses/>.
```

## Contents

This repository contains `spotrod`, a semi-analytic model for transits of spotted stars. The model is implemented in C and comes with Python API. The following files and directories are included in the root directory:

- [toyproblem](toyproblem) A toy problem benchmarking Python, Cython, and C implementations of circleangle(), the function calculating $\beta$;
- [COPYING](COPYING) the GNU General Public License;
- [Makefile](Makefile) makefile for generating toycython.so and toyc.so;
- [README.md](README.md) this file;
- [spotrod.c](spotrod.c) C implementation of the model;
- [spotrod.h](spotrod.h) headers for functions in spotrod.c;
- [spotrod-python-wrapper.c](spotrod-python-wrapper.c) C code wrapping spotrod.c in the numpy C API;
- [spotrod-setup.py](spotrod-setup.py) a Python script for compiling spotrod.so;
- [test.py](test.py) a minimal script generating a model lightcurve;
- [test.png](test.png) output of minimal script;
- [mcmc.py](mcmc.py) an MCMC simulation using the [emcee](http://dan.iel.fm/emcee/) package, creating an animation.

## Compilation

To generate the module `spotrod.so`, run `make` without any arguments.

## Citation

If you use `spotrod` in a publication, please consider citing [Béky, Kipping, and Holman, 2014, arXiv:1407.4465](http://adsabs.harvard.edu/abs/2014arXiv1407.4465B).
