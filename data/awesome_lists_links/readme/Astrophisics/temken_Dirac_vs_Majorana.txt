[![Build Status](https://travis-ci.com/temken/Dirac_vs_Majorana.svg?branch=master)](https://travis-ci.com/temken/Dirac_vs_Majorana)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)


# Dirac vs Majorana dark matter

<a href="https://ascl.net/2112.012"><img src="https://img.shields.io/badge/ascl-2112.012-blue.svg?colorB=262255" alt="ascl:2112.012" /></a>
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3701262.svg)](https://doi.org/10.5281/zenodo.3701262)
[![arXiv](https://img.shields.io/badge/arXiv-2003.04039-B31B1B.svg)](https://arxiv.org/abs/2003.04039)

Statistical discrimination of sub-GeV Majorana and Dirac dark matter at direct detection experiments.

<img src="https://user-images.githubusercontent.com/29034913/76204669-209afa80-61f9-11ea-9cbc-3481bada2e1c.png" width="800">

## GENERAL NOTES

Direct detection experiments which look for sub-GeV dark matter (DM) particles often search for DM-induced electronic transitions inside a target. Assuming one of these experiments would succeed, the next question would be to study the properties of DM.

One question we could ask is if DM particles are their own anti-particles, a so-called Majorana fermion. This code determines the statistical significance with which a successful electron scattering experiment could reject the Majorana hypothesis (using the likelihood ratio test) in favour of the hypothesis of Dirac DM. We assume that the DM interacts with the photon via higher-order electromagnetic moments.

> :warning: **Warning**: In order for this code to produce results, the */data/* folder needs to contain the tabulated atomic response functions, which can be computed with the [DarkARC tool](https://github.com/temken/DarkARC). Without these tables, it is not possible to compute ionization spectra and predictions for signal event rates.

## CONTENT

- */bin/*: Contains the executable.
- */build/*: Will contain the object files after compilation.
- */data/*: The folder for the tabulated atomic response functions.
- */include/*: All the header files can be found here.
- */results/*: Resulting tables and files are saved to this folder.
- */src/*: Contains the source code.

## Installation:

The code can be compiled using the makefile. It might be necessary to adjust the compiler lines and the path to the libraries:

```
#Compiler and compiler flags
CXX := g++
CXXFLAGS := -Wall -std=c++11 
LIB := 
INC := -I include
(...)
```

The code is compiled by running 
```
make
```
from the root directory in the terminal to compile DiracVsMajorana.

Running
```
make clean
```
deletes all object files and executables.


## CITING THIS CODE

If you decide to use this code, please cite the latest archived version,

> [[DOI:10.5281/zenodo.3701262]](https://doi.org/10.5281/zenodo.3701262)

as well as the original publications,

>Catena, R., Emken, T. , Ravanis J., **Rejecting the Majorana nature of dark matter with electron scattering experiments**, [[arXiv:2003.04039]](https://arxiv.org/abs/2003.04039).

## VERSIONS

- **v1.0** (06/03/2020): Version released with v1 of the preprint [[arXiv:2003.04039v1]](https://arxiv.org/abs/2003.04039v1).

## AUTHORS & CONTACT

The author of this code is Timon Emken.

For questions, bug reports or other suggestions please contact [emken@chalmers.se](mailto:emken@chalmers.se).


## LICENCE

This project is licensed under the MIT License - see the LICENSE file.
