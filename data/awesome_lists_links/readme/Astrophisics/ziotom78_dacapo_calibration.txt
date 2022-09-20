# Calibrating TODs using the DaCapo algorithm

[![ascl:1612.007](https://img.shields.io/badge/ascl-1612.007-blue.svg?colorB=262255)](http://ascl.net/1612.007)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the implementation of the calibration
algorithm used in the
[Planck](http://www.cosmos.esa.int/web/planck)/LFI
[2015 data release](http://www.cosmos.esa.int/web/planck/pla). It is a
Python3/Fortran program developed using
[literate programming techniques](https://en.wikipedia.org/wiki/Literate_programming),
and it uses MPI to distribute the workload among several processing
units.

The full source code of the program can be found in the PDF document
[dacapo_calibration.pdf](https://github.com/ziotom78/dacapo_calibration/blob/master/dacapo_calibration.pdf).

## Installation

After you have downloaded the repository, just type `make`. You can
configure the programs `make` calls by setting variables in a new
file named `configuration.mk`. The following variables are
accepted:
- `NOWEAVE`
- `NOTANGLE`
- `CPIF`
- `TEX2PDF`
- `BIBTEX`
- `PYTHON`
- `F2PY`
- `AUTOPEP8`
- `DOCKER`
- `MPIRUN`
- `INKSCAPE`
- `MV`

## Example

In the directory `examples` you will find a couple of parameter files
for `index.py` and `calibrate.py`. Refer to the PDF documentation for
a full reference of each parameter.

## Testing

To run a number of unit tests, run the following command:

    make check

To run a number of integration test, use the following command instead:

    make fullcheck

## License

The code is released under a permissive MIT license. See the file
[LICENSE](https://github.com/ziotom78/dacapo_calibration/blob/master/LICENSE).

## Citation

If you use this code in your publications, please cite it using the following BibTeX entry:
`````
@MISC{2016ascl.soft12007T,
   author = {{Tomasi}, M.},
    title = "{dacapo\_calibration: Photometric calibration code}",
 keywords = {Software },
howpublished = {Astrophysics Source Code Library},
     year = 2016,
    month = dec,
archivePrefix = "ascl",
   eprint = {1612.007},
   adsurl = {http://adsabs.harvard.edu/abs/2016ascl.soft12007T},
  adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

`````
