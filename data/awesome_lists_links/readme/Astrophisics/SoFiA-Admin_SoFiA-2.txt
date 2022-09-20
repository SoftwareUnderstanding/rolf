# SoFiA 2

[![Docker build latest](https://github.com/axshen/SoFiA-2/actions/workflows/latest-image-build.yml/badge.svg)](https://github.com/axshen/SoFiA-2/actions/workflows/latest-image-build.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

This is version 2 of the HI Source Finding Application (SoFiA), a **source finding pipeline** originally designed to detect and characterise galaxies in 3D extragalctic HI data cubes. SoFiA 2 is a reimplementation of the original SoFiA pipeline in the C programming language and provides most of the functionality of SoFiA 1.x. While some development is still happening, SoFiA 2 is stable and can be used in production mode at this point. Although we strongly recommend switching to SoFiA 2, users will still be able to use **SoFiA 1.x** (https://github.com/SoFiA-Admin/SoFiA) for processing their data.


## Improvements in SoFiA 2

* Being written in C and making extensive use of **multi-threading**, SoFiA 2 is much faster than SoFiA 1.x.
* SoFiA 2 requires significantly **less memory** than SoFiA 1.x (down from > 5 × cube size to ~ 2.3 × cube size).
* SoFiA 2 currently has only a **single external dependency** (wcslib) and should therefore compile and run on any machine with a Linux or Unix operating system and the GCC compiler and wcslib installed.
* A wrapper called **SoFiA-X** (https://github.com/AusSRC/SoFiAX) is available for distributed processing on multiple HPC nodes.


## Prerequisites

The following software is required before SoFiA 2 can be installed:

* **Linux** or **Unix** operating system (e.g. Ubuntu, Mac OS, etc.)
* **GNU C compiler** (gcc) version 4 or higher (https://gcc.gnu.org/)
* **wcslib** version 7 or higher (https://www.atnf.csiro.au/people/mcalabre/WCS/)

Both gcc and wcslib are freely available under the GNU General Public Licence. Note that earlier versions of gcc or wcslib might work as well, but this has not been tested. In principle, other compilers that are compatible with gcc and support the C99 standard might also work, e.g. Apple’s clang compiler or the Intel C compiler, possibly with some minor tweaking of the installation script.

You may want to first check if wcslib is either already installed or available from your operating system’s software repository (`wcslib-dev` package on Ubuntu/Debian; `wcslib-devel` on Fedora/CentOS/Red Hat; `wcslib` on MacOS/Homebrew) before attempting to install it manually.

## Installation

Once all prerequisites are installed and available, simply enter the SoFiA 2 base directory and execute the `compile.sh` script to compile SoFiA 2 using the GCC compiler:

`./compile.sh -fopenmp`

Note that the `-fopenmp` parameter is optional and will enable **multi-threading** using OpenMP. If your compiler does not support OpenMP, this parameter can simply be omitted to install a single-threaded version of SoFiA 2. Please ensure that you read and follow the **instructions** printed at the end of the compilation process to finalise the installation. If a compiler error related to WCSLIB shows up, please ensure that WCSLIB is installed in a standard location where it can be found by the GCC. Then run the compilation script again to see if the error message has disappeared. If for some reason you don’t have permission to execute the compilation script, try running the command `chmod 764 compile.sh` first to set the correct file permissions.

As an alternative, we provide a `Makefile` for those who prefer to use `make` to install SoFiA 2. The `Makefile` itself contains a few examples of how to invoke it with different compilers (with or without OpenMP support). Note that if you prefer to use `make`, you may still want to create a symbolic link or alias to the `sofia` executable file in the end to make SoFiA 2 easily accessible across your system.

**NOTE:** Others may have created alternative ways of downloading and installing SoFiA 2. As we have no control over such third-party distributions or packages, we cannot provide any support for installation methods other than the one described here. If you have installed SoFiA 2 through a third-party repository, please contact the administrator of that package in the case of installation issues.

### Docker containers

We also provide official Docker containers for all stable releases of SoFiA 2. These are available from https://hub.docker.com/r/sofiapipeline/sofia2. Note that Docker images can also be run in Singularity (see https://sylabs.io/guides/latest/user-guide/singularity_and_docker.html). However, due to the significant overhead imposed by pre-packaged containers, we strongly advise users to install SoFiA 2 from source, which is straightforward using the instructions above and will use significantly less disc space.


## Documentation

An overview of all control parameters as well as a PDF copy of the SoFiA 2 User Manual can be found on the wiki at https://github.com/SoFiA-Admin/SoFiA-2/wiki. The wiki also contains a small test data cube and parameter file that users can download to verify their SoFiA 2 installation.


## Additional tools

Several useful tools have been developed to help with running SoFiA 2 or process the output produced by the pipeline:

* **SoFiA-X** is a parallel wrapper around SoFiA 2 that can be used to process large data cubes in a supercomputing environment.
  Download: https://github.com/AusSRC/SoFiAX
* **OptiFind** is a wrapper around SoFiA 2 that can be used to search for HI emission in small regions of a data cube as defined by an input catalogue.
  Download: https://github.com/SoFiA-Admin/OptiFind
* **SoFiA Image Pipeline** is a powerful Python script for generating plots of SoFiA output products, including images, spectra and PV diagrams.
  Download: https://github.com/kmhess/SoFiA-image-pipeline
* **SpecPlot** is a tool for producing publication-ready plots of integrated spectra from SoFiA 2.
  Download: https://github.com/SoFiA-Admin/SpecPlot
* **BusyFit** is a tool for fitting the Busy Function to an integrated spectrum for the purpose of accurate parameterisation.
  Download: https://github.com/SoFiA-Admin/BusyFit


## Citation

Users are requested to cite the following two papers in any work that is based on the use of SoFiA 2:

* **SoFiA: a flexible source finder for 3D spectral line data**  
  Serra, P., Westmeier, T., Giese, N., et al., 2015, MNRAS, 448, 1922  
  https://ui.adsabs.harvard.edu/abs/2015MNRAS.448.1922S

* **SoFiA 2 – An automated, parallel HI source finding pipeline for the WALLABY survey**  
  Westmeier, T., Kitaeff, S., Pallot, D., et al., 2021, MNRAS, 506, 3962  
  https://ui.adsabs.harvard.edu/abs/2021MNRAS.506.3962W


## Feedback

Should you decide to run SoFiA 2 on your own data cubes, we would welcome any feedback on how well SoFiA 2 works for you and whether any improvements could be made. If you have a GitHub account, you can directly create a new issue (https://github.com/SoFiA-Admin/SoFiA-2/issues/new) on GitHub for questions, feature requests or bug reports. Alternatively, please feel free to directly contact the project leader, Tobias Westmeier, via e-mail at `tobias.westmeier (at) uwa.edu.au` to provide feedback on your experience with SoFiA 2. Note that the main purpose of SoFiA 2 is to facilitate the processing of HI data from SKA precursor surveys, and we are unlikely to be able to accommodate requests for additional features beyond this primary scope.


## Copyright and licence

Copyright (C) 2022 The SoFiA 2 Authors

SoFiA 2 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License  along with this program. If not, see http://www.gnu.org/licenses/.
