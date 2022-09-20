
# WARNING: THIS REPO IS NO LONGER MAINTAINED

**A new major revision of the software has been developed and is available at 
https://github.com/bek0s/gbkfit**

# GBKFIT

GBKFIT is a high-performance open-source software for modelling galaxy
kinematics from 3D spectroscopic observations. 

## Installation guide

For instructions on how to install GBKFIT read [here](INSTALL.md).

GBKFIT is a new software and thus it has not been tested in many different 
platforms. If you find problems or bugs related to the installation process 
please send me an e-mail or create a new GitHub issue. Your help is greatly 
appreciated. 

## User guide

For instructions on how to use GBKFIT read [here](USERGUIDE.md).

## Credits

GBKFIT is developed by me (Georgios Bekiaris) during my PhD studies at the 
Swinburne University of Technology and my visit to University of Toronto.

If you use GBKFIT in a publication please cite:
[Bekiaris et al. 2016](http://adsabs.harvard.edu/abs/2016MNRAS.455..754B).

## A brief introduction

GBKFIT is a high-performance open-source software for modelling galaxy
kinematics from 3D spectroscopic observations. It is written in C++, and uses
the CMake build system.

GBKFIT features a modular architecture which allows it to use a variety of
data models, galaxy models, and optimization techniques. It also provides a
clean object-oriented interface which enables programmers to create and add
their own custom models and optimization techniques into the software.

GBKFIT models observations with a combination of two models: a Data Model
(DModel), and a Galaxy Model (GModel). The former is used to describe the data
structure of the observation, while the latter is used to describe the
observed galaxy. By convention, the name of the data and galaxy models start 
with `gbkfit.dmodel.` and `gbkfit.gmodel.` respectively.

In GBKFIT, the optimization techniques are called fitters, and by convention, 
their names start with `gbkfit.fitter.`.

### Performance

Galaxy kinematic modelling is a computationally intensive process and it can
result very long run times. GBKFIT tackles this problem by utilizing the
many-core architectures of modern computers. GBKFIT can accelerate the
likelihood evaluation step of the fitting procedure on the Graphics Processing
Unit (GPU) using CUDA. If there is no GPU available on the system, it can use
all the cores available on the Central Processing Unit (CPU) through OpenMP.

### Data models

GBKFIT comes with the following data models:
- `gbkfit.dmodel.mmaps_<device_api>`: This model is used to describe moment
maps extracted from a spectral data cube. Thus, this model should be used to
perform 2D fits to velocity and velocity dispersion maps. Flux maps are also
supported but they are currently experimental and should not be used.
- `gbkfit.dmodel.scube_<device_api>`: This model is used to describe spectral
data cubes. Thus, this model should be used to perform 3D fits to spectral
data cubes. Support for 3D fitting is experimental and should be avoided for
now.

`<device_api>` can be either `omp` (for multi-threaded CPU acceleration) or
`cuda` (for GPU acceleration).

### Galaxy models

GBKFIT comes with the following galaxy models:
- `gbkfit.gmodel.gmodel1_<device_api>`: This model is a combination of a thin
and flat disk, a surface brightness profile, a rotation curve, and an intrinsic
velocity dispersion which is assumed to be constant across the galactic disk.

  The following surface brightness profiles are supported:
  - Exponential disk

  The following rotation curve profiles are supported:
  - Linear ramp
  ([Wright et al. 2007](http://adsabs.harvard.edu/abs/2007ApJ...658...78W))
  - Arctan
  ([Courteau 1997](http://adsabs.harvard.edu/abs/1997AJ....114.2402C))
  - Boissier
  ([Boissier et al. 2003](http://adsabs.harvard.edu/abs/2003MNRAS.346.1215B))
  - Epinat
  ([Epinat et al. 2008](http://adsabs.harvard.edu/abs/2008MNRAS.388..500E))

`<device_api>` can be either `omp` (for multi-threaded CPU acceleration) or
`cuda` (for GPU acceleration).

### Fitters

GBKFIT comes with the following fitters:
- `gbkfit.fitter.mpfit`: This fitter employs the Levenberg-Marquardt Algorithm
through the [MPFIT](https://www.physics.wisc.edu/~craigm/idl/cmpfit.html)
library.
- `gbkfit.fitter.multinest`: This fitter employs a modified version of the 
Nested Sampling technique through the
[MultiNest](https://ccpforge.cse.rl.ac.uk/gf/project/multinest/) library.


### Point Spread Functions

GBKFIT supports the following Point Spread Function (PSF) models: 2D
elliptical Gaussian, 2D elliptical Lorentzian, and 2D elliptical Moffat.
Alternatively, the user can supply a 2D image.

### Line Spread Functions

GBKFIT supports the following Line Spread Function (LSF) models: 1D Gaussian,
1D Lorentzian, and 1D Moffat. Alternatively, the user can supply an 1D image.

