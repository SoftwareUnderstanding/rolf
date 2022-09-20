## EQUIB
[![Build Status](https://travis-ci.com/equib/EQUIB.svg?branch=master)](https://travis-ci.com/equib/EQUIB)
[![Binder](http://mybinder.org/badge.svg)](http://mybinder.org/repo/equib/equib)
[![GitHub license](https://img.shields.io/badge/license-GPL-blue.svg)](https://github.com/equib/EQUIB/blob/master/LICENSE)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281/zenodo.2584191-blue.svg)](https://doi.org/10.5281/zenodo.2584191)
[![ASCL](https://img.shields.io/badge/ASCL-1603.005-green.svg)](http://adsabs.harvard.edu/abs/2016ascl.soft03005H)

**Program EQUIB**  (FORTRAN 90)

The Fortran program EQUIB solves the statistical equilibrium equation for each ion and yields atomic level populations and line emissivities for given physical conditions, namely electron temperature and electron density, appropriate to the zones in an ionized nebula where the ions are expected to exist.

### Installation

How to compile:

    make

How to clean:

    make clean

### Usage

How to perform **plasma diagnostics**:

    ./diagnostic_equib arg1 arg2 arg3 arg4 arg5 arg6

*arg1*: the ion name e.g. 's_ii'.

*arg2*: the upper level from [lines levels](https://github.com/equib/EQUIB/blob/master/docs/lineslevels.readme) e.g. '1,2,1,3/' or '1,2/'.

*arg3*: the lower level from [lines levels](https://github.com/equib/EQUIB/blob/master/docs/lineslevels.readme) e.g. '1,5/' or '1,3/'.

*arg4*: the diagnostics line flux ratio of density- or temperature-sensitive lines e.g. '10.753'.

*arg5*: the diagnostics type: 'T' for temperature and 'D' for density.

*arg6*: the value of another fixed physical condition (density or temperature). It is density if arg5 = 'T', and temperature if arg5 = 'D'.

For example, to determine the electron temperature:

    ./diagnostic_equib 's_ii' '1,2,1,3/' '1,5/' '10.753' 'T' '2550.0'

For example, to determine the electron density:

    ./diagnostic_equib 's_ii' '1,2/' '1,3/' '1.506' 'D' '7000.0'

How to perform **abundance analysis**:

    ./abundance_equib arg1 arg2 arg3 arg4 arg5

*arg1*: the ion name e.g. 'o_iii'.

*arg2*: the atomic level from [lines levels](https://github.com/equib/EQUIB/blob/master/docs/lineslevels.readme) e.g. '3,4/'.

*arg3*: the electron temperature.

*arg4*: the electron density.

*arg5*: the observed line flux relative to Hb, on a scale where I(Hb)=100.0

For example, to determine the O2+ abundance from [O III] line flux I([O III])=1200, where Te=10000 K, and Ne=5000 cm-3:

    ./abundance_equib 'o_iii' '3,4/' '10000.0' '5000.0' '1200.0'

### References

* I.D. Howarth, S. Adams, et al., [Astrophysics Source Code Library, ascl:1603.005, 2016](http://adsabs.harvard.edu/abs/2016ascl.soft03005H)

* I.D. Howarth & S. Adams, [University College London, 1981](http://adsabs.harvard.edu/abs/1981ucl..rept.....H)

* I.D. Howarth & B. Wilson, [MNRAS, 204, 1091, 1983](http://adsabs.harvard.edu/abs/1983MNRAS.204.1091H)

* S. Adams & M.J. Seaton, [MNRAS, 200, 7P, 1982](http://adsabs.harvard.edu/abs/1982MNRAS.200P...7A)

* J.P. Harrington, M.J. Seaton, S. Adams & J.H. Lutz, [MNRAS, 199, 517, 1982](http://adsabs.harvard.edu/abs/1982MNRAS.199..517H)

* J.P. Harrington, [ApJ, 152, 943, 1968](http://adsabs.harvard.edu/abs/1968ApJ...152..943H)

* J.P. Harrington, [Ph.D.Thesis, Ohio State University, 1967](http://adsabs.harvard.edu/abs/1967PhDT.........6H)

### Publications used EQUIB

The Program EQUIB has been widely used in around 50 peer-reviewed publications since 1984: 

* Citations for the Program EQUIB from the [ADS Databases](http://adsabs.harvard.edu/cgi-bin/nph-ref_query?bibcode=1981ucl..rept.....H&amp;refs=CITATIONS)

### Acknowledgement

If you employ the Program EQUIB in your work, please acknowledge the usage by citing the following references:
	
* I. D. Howarth, S. Adams, R. E. S. Clegg, D. P. Ruffle, X.-W. Liu, C. J. Pritchet, & B. Ercolano, Astrophysics Source Code Library,  [ascl:1603.005](http://adsabs.harvard.edu/abs/2016ascl.soft03005H), 2016

* I.D. Howarth & S. Adams, [Program EQUIB](http://adsabs.harvard.edu/abs/1981ucl..rept.....H), University College London, 1981
