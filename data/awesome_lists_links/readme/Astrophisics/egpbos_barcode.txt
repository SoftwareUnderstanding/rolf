# Barcode
Bayesian Reconstruction of COsmic DEnsity fields

This repository contains both Barcode and a set of supplementary analysis tools.

[![Build Status](https://travis-ci.org/egpbos/barcode.svg?branch=master)](https://travis-ci.org/egpbos/barcode)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/db7aa18754fa4720a2d80ab47ed85e3b)](https://www.codacy.com/app/egpbos/barcode?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=egpbos/barcode&amp;utm_campaign=Badge_Grade)
[![Coverage Status](https://coveralls.io/repos/github/egpbos/barcode/badge.svg?branch=master)](https://coveralls.io/github/egpbos/barcode?branch=master)
[![Join the chat at https://gitter.im/barcode_cosmo/community](https://badges.gitter.im/barcode_cosmo/community.svg)](https://gitter.im/barcode_cosmo/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

## Citing

If you use this software, please cite it as:

Bos E. G. P., Kitaura F.-S., van de Weygaert R., 2019, [MNRAS](http://dx.doi.org/10.1093/mnras/stz1864), 488, 2573

In bibtex format:

```bibtex
@ARTICLE{2019MNRAS.488.2573B,
       author = {{Bos}, E.~G. Patrick and {Kitaura}, Francisco-Shu and
                 {van de Weygaert}, Rien},
        title = "{Bayesian cosmic density field inference from redshift space dark matter maps}",
      journal = {MNRAS},
     keywords = {methods: analytical, methods: statistical, galaxies: distances and redshifts, cosmology: observations, large-scale structure of Universe, Astrophysics - Cosmology and Nongalactic Astrophysics, Statistics - Applications, Statistics - Computation},
         year = "2019",
        month = "Sep",
       volume = {488},
       number = {2},
        pages = {2573-2604},
          doi = {10.1093/mnras/stz1864},
archivePrefix = {arXiv},
       eprint = {1810.05189},
 primaryClass = {astro-ph.CO},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2019MNRAS.488.2573B},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

Unique identifiers for citing the software itself (preferably in addition to citing the paper above) are provided through Zenodo (a unique DOI for each Barcode release) and the Astrophysics Source Code Library (all Barcode versions).

[![DOI](https://zenodo.org/badge/152633059.svg)](https://zenodo.org/badge/latestdoi/152633059) 
<a href="http://ascl.net/1810.002"><img src="https://img.shields.io/badge/ascl-1810.002-blue.svg?colorB=262255" alt="ascl:1810.002" /></a>

This code was previously described and applied in [conference proceedings](https://arxiv.org/abs/1611.01220) and in [Patrick's PhD thesis](https://www.rug.nl/research/portal/en/publications/clusters-voids-and-reconstructions-of-the-cosmic-web(0f7c3d17-9661-4b9f-a27c-dfac2990b844).html).

## Build

### Install dependencies
Before compiling, make sure you have the required dependencies:

- CMake
- A compiler supporting at least C++11 (e.g. gcc 7 or clang 5)
- FFTW 3
- GNU Scientific Library
- ncurses

If you install these using a Linux package manager, make sure you get the development versions of the packages, i.e. the ones ending in `-dev` (`libfftw3-dev`, etcetera).
For instance, with apt-get in Debian or Ubuntu, you can install the requirements with:

```sh
sudo apt-get install cmake g++ fftw3-dev libgsl-dev libncurses-dev
```

Using MacPorts on macOS, you can install the necessary packages with:

```sh
sudo port install cmake fftw-3 fftw-3-single gsl ncurses
```

You can also use `conda` to install everything in a virtual environment:

```sh
conda create -n barcode cmake cxx-compiler fftw gsl ncurses -c conda-forge
```

When using the `conda` environment, make sure you activate it before compiling and using `barcode`:

```sh
conda activate barcode
```

### Compile the code

Clone the repository and `cd` into the cloned directory:
```sh
git clone https://github.com/egpbos/barcode.git
cd barcode
```

Then run `cmake` and `make` to configure and build:

```sh
mkdir cmake-build
cd cmake-build
cmake ..
make
```

This will create binaries for barcode and the supplementary tools in the `cmake-build` directory.


## Run

Barcode must be run in the same directory as the `input.par` file.
Edit this file to change input parameters.
Then simply run with:

```
cmake-build/barcode [restart_iteration]
```

Optionally add the `restart_iteration` number when doing a restart run from existing output files.


## Development and contributing
This is an early release. Unit tests and other test codes will be added later (as mentioned in some of the code comments). Documentation as well.

Contributions are very welcome! Please don't hesitate to propose ideas for improvements in the GitHub issues or in a PR.


## License
The original contributions made as part of this code are distributed under the MIT license (see `LICENSE` file).

When compiled, this code must link to FFTW 3 and the GNU Scientific Library (GSL).
FFTW is distributed under the GNU Public License v2 or a later version, GSL under GPL v3.
This means that any redistribution of Barcode in binary form is subject to GPL v3 terms.

<!-- Barcode also depends on ncurses, which is distributed under the X11 license, a permissive license similar to the MIT license. -->
