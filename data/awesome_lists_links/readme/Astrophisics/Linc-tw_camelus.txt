Camelus
=======

Counts of Amplified Mass Elevations from Lensing with Ultrafast Simulation  
Chieh-An Lin (IfA Edinburgh)  
Release v2.0 - 2018-03-14  
[<p align="center"><img src="http://www.cosmostat.org/wp-content/uploads/2014/11/Logo_Camelus_fig_name_vertical.png" width="240px" /></p>](http://species.wikimedia.org/wiki/Camelus)


Description
-----------

Camelus is a fast weak-lensing peak-count modeling algorithm in C. It provides a prediction on peak counts from input cosmological parameters.

Here is the summary of the algorithm:
- Sample halos from a mass function
- Assign density profiles, randomize their positions
- Compute the projected mass, add noise
- Make maps and create peak catalogues

For a more detailed description, please take a look at [Lin & Kilbinger (2015a)](http://arxiv.org/abs/1410.6955).


Requirements
------------

Required softwares:
- [cmake](https://cmake.org/cmake/resources/software.html)
- [pkg-config](https://www.freedesktop.org/wiki/Software/pkg-config/)
- [gcc](https://gcc.gnu.org/)
- [gsl](https://www.gnu.org/software/gsl/)
- [fftw](https://www.fftw.org/)
- [nicaea v2.7](http://www.cosmostat.org/nicaea.html)

Optional softwares:
- [cfitsio](https://heasarc.gsfc.nasa.gov/fitsio/fitsio.html)
- [chealpix](https://healpix.jpl.nasa.gov/index.shtml)
- [healpix_cxx](https://healpix.jpl.nasa.gov/index.shtml)
- [openmpi](https://www.open-mpi.org/)

During the compilation, cmake uses pkg-config to find optional softwares. If they are missing, the compilation still continues without providing all functionalitites.


Compilation
-----------

For Mac users, do the follows before compilation:
```Bash
$ export CC=gcc
$ export CXX=g++
```
or use `setenv` command in tcsh.

To compile the package:
```Bash
$ export NICAEA={PATH_OF_NICAEA2.7}
$ cd build
$ cmake ..
$ make
```

To get program instructions:
```Bash
$ ./camelus
```


Updates
-------

Current release: Camelus v2.0

##### New features in v2.0 - Mar 14, 2018
- Updated consistency with Nicaea to v2.7
- Flexible compilation: missing optional packages will not stop the compilation
- Renamed peakParameters.c/.h into parameters.c/.h
- Renamed constraint.c/.h into multiscale.c/.h
- Splited rayTracing.c/.h into galaxySampling.c/.h and rayTracing.c/.h
- New files: FITSFunctions.c/.h
- New files: HEALPixFunctions.c/.h
- Flexible parameter reading mechanism
- More explanatory peakParam.par file
- Allowed output customization
- Renamed confusing parameter names
- Random halo redshift inside each slice instead of the median value
- Allowed random mask for plat geometry
- Allowed HEALPix geometry
- Allowed galaxy weighting
- Allowed the mass-sheet correction of lensing signals
- Allowed the Seitz & Schneider inversion method
- Allowed both local and global noise levels in the S/N
- Removed nonlinear filtering
- Added the license

##### New features in v1.31 - Mar 22, 2016:
- Made installation more friendly by removing the dependency on cfitsio and mpi
- Added the routine for computing 1-halo & 2-halo terms of the convergence profile
- Flexible parameter space for PMC ABC
- Remove files: FITSFunctions.c/.h

##### New features in v1.3 - Dec 09, 2015:
- New files: constraint.c/.h
- Allowed multiscale peaks in one data vector
- Allowed a data matrix from several realizations
- Used the local galaxy density as the noise level in the S/N
- Increased the parameter dimension for PMC ABC
- Changed the summary statistic options for PMC ABC

Unavailable features because of the exterior file dependency:
- New files: FITSFunctions.c/.h
- Added the mask option and nonlinear filtering

##### New features in v1.2 - Apr 06, 2015:
- Improved the computation speed by a factor of 6~7
- Converted the halo array structure into a binned structure, called "halo_map"
- Converted the galaxy tree structure into a binned structure, called "gal_map"
- New files: ABC.c/.h
- Added the population Monte Carlo approximate Bayesian computation (PMC ABC) algorithm

##### New features in v1.1 - Jan 19, 2015:
- Fixed the bug from calculating halo radii

##### New features in v1.0 - Oct 24, 2014:
- Fast weak lensing peak count modeling


License
-------

Camelus is distributed under the terms of the [GNU General Public License Version 3 (GPLv3)](https://www.gnu.org/licenses/).

The license gives you the option to distribute your application if you want to. You do not have to exercise this option in the license.

If you want to distribute an application which uses Camelus, you need to use the GNU GPLv3.


References
----------

- [Baltz et al. (2009)](https://arxiv.org/abs/0705.0682) - JCAP, 1, 15
- [Bartelmann & Schneider (2001)](https://arxiv.org/abs/astro-ph/9912508) - Phys. Rep., 340, 291
- [Fan et al. (2010)](https://arxiv.org/abs/1006.5121) - ApJ, 719, 1408
- [Hetterscheidt et al. (2005)](https://arxiv.org/abs/astro-ph/0504635) - A&A, 442, 43
- [Lin & Kilbinger (2015a)](https://arxiv.org/abs/1410.6955) - A&A, 576, A24
- [Lin & Kilbinger (2015b)](https://arxiv.org/abs/1506.01076) - A&A, 583, A70
- [Lin et al. (2016)](https://arxiv.org/abs/1603.06773) - Submitted to A&A
- [Marin et al. (2011)](https://arxiv.org/abs/1101.0955)
- [Oguri & Hamana (2011)](https://arxiv.org/abs/1101.0650) - MNRAS, 414, 1851
- [Seitz & Schneider (1995)](https://arxiv.org/abs/astro-ph/9408050) - A&A, 297, 287
- [Takada & Jain (2003a)](https://arxiv.org/abs/astro-ph/0209167) - MNRAS, 340, 580
- [Takada & Jain (2003b)](https://arxiv.org/abs/astro-ph/0304034) - MNRAS, 344, 857
- [Weyant et al. (2013)](https://arxiv.org/abs/1206.2563) - ApJ, 764, 116
- Wright & Brainerd (2000) - ApJ, 534, 34


Contact information
-------------------

Author:
- [Chieh-An Lin](https://linc.tw/)

Contributors:
- [Martin Kilbinger](http://www.cosmostat.org/people/kilbinger/)
- [François Lanusse](https://flanusse.net/)

Please feel free to send questions, feedback and bug reports to calin (at) roe.ac.uk.  
Check also the [GitHub repository](https://github.com/Linc-tw/camelus) of the code.

