## EarthScatterLikelihood

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3725882.svg)](https://doi.org/10.5281/zenodo.3725882) [![arXiv](https://img.shields.io/badge/arXiv-2004.01621-B31B1B)](https://arxiv.org/abs/2004.01621) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Last Update:** 06/10/2021

Code for calculating event rates and likelihoods for Earth-scattering DM, released in association with the paper "*Measuring the local Dark Matter density in the laboratory*" ([arXiv:2004.01621](https://arxiv.org/abs/2004.01621)). The code is written in Fortran (for some reason), with plotting routines in Python.

The folder [`DaMaSCUS_results/`](DaMaSCUS_results/) contains results from Monte Carlo simulations, which are read in by the code. These were generated using [`DaMaSCUS`](https://github.com/temken/DaMaSCUS/tree/v1.1).

The event rates and likelihood calculators currently work in the ranges:
* m_x = [0.058, 0.5] GeV  
* sigma_p^SI = [1e-40, 1e-30] cm^2.

#### Getting started

You can compile the EarthScatterLikelihood code using the Makefile in the [`src/`](src/) folder. Simply run `Make` from that folder and it will compile the `calcContour` binary. To run parameter reconstructions, you can then do:
```
./calcContour M_X SIGMA_B DATA FIX_MASS LAT_DET OUTPATH
```

The command line arguments are as follows:
* `M_X` - WIMP mass in GeV  
* `SIGMA_B` - Benchmark WIMP-proton cross section in cm^2  
* `DATA` - flag for which data to use (1 = Energy + time, 2 = time only, 3 = energy only)  
* `FIX_MASS` - flag for whether the WIMP mass should be kept fixed (1 = fix to benchmark value, 0 = profile in range [0.058, 0.5] GeV)  
* `LAT_DET` - detector latitude (in degrees, over the range [-90, 90])  
* `OUTPATH` - output folder to save results to (this will be `results/OUTPATH/`)

An example is in `GetContours.sh`. Beware that with the current settings (number of grid points to scan over), these reconstructions may take some time.

The files `RunMPI_Contours.py` and `SubmitSims.sh` may be useful for submitting many reconstructions in parallel on a cluster.

You can also use the makefile to compile some other code, such as event generators and generic likelihood calculators, which were not used in the main contour calculations, but which might be useful. See [`src/`](src/) for more details.

You can edit the properties of the detector by editing the [`src/expt.f90`](src/expt.f90) file before compiling. Some examples are in [`src/expt_EDE.f90`](src/expt_EDE.f90) and [`src/expt_SAPPHIRE.f90`](src/expt_SAPPHIRE.f90) for a Germanium and a Sapphire detector respectively.

#### Plots and plotting

The [`plotting/`](plotting/) folder contains some plotting routines in python for loading in the p-value data and generating plots from the paper. Note that in order to reproduce the results of the paper "*Measuring the local Dark Matter density in the laboratory*" ([arXiv:2004.01621](https://arxiv.org/abs/2004.01621)), you will also need the tabulated p-values, which were generated with the EarthScatterLikelihood code: [DOI:10.5281/zenodo.3739341](https://doi.org/10.5281/zenodo.3739341). **NOTE: These tabulated p-values are currently out of date and will be updated shortly**. Simply extract those data files into the [`results/`](results/) folder and you should be able to use the plotting routines.