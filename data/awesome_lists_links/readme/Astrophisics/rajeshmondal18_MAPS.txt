# MAPS
The Multi-frequency Angular Power Spectrum (MAPS) estimator

Documentation
=============
The user is referred to section 3.2 of Mondal et al. 2018 (https://arxiv.org/abs/1706.09449) for a detailed description of the algorithm.

Requirements
============
The code starts by reading a gridded data in cartesian (x, y, z) coordinates or in (theta_x, theta_y, nu) coordinates.

You need to install FFTW-3.x.x with the following flags: '--enable-float', '--enable-threads' and '--enable-openmp' to compile this set of codes. Look at the installation instruction http://www.fftw.org/fftw3_doc/Installation-on-Unix.html#Installation-on-Unix

Compilation and Run
===================
Use the makefile for compilation in the following manner:

$ make

It will create the executable 'MAPS'.

In the 'Input Parameters' section of the code, you need to specify

1. The number of angular multipole (l) bins
2. Grid spacing in Mpc
3. Delta-theta
4. Delta-nu in MHz

according to your need.

An input data "LC8" (binary data) has been provided here https://drive.google.com/file/d/1d-HxJvbzhs0I33YBL6_fuDUFCz24ArNw/view?usp=sharing for testing.

To run:

$ ./MAPS

Acknowledging
=============
If you are using this code, please consider acknowledging the following papers.

1. Mondal et al. 2018 https://doi.org/10.1093/mnras/stx2888
2. Mondal et al 2019 https://doi.org/10.1093/mnrasl/sly226
3. Mondal et al. 2020 https://doi.org/10.1093/mnras/staa1026
4. Mondal et al 2021
