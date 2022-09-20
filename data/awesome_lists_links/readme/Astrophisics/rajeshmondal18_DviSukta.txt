# DviSukta
A direct estimator of the bispectrum
_____________________________________

This is a parallelized code for the calculation of the Spherically Averaged Bispectrum (SABS). It is based on a new optimised direct estimation method and written in C.
_____________________________________

The user is referred to section 2 of Mondal et al. 2021 (https://arxiv.org/abs/2107.02668) for a detailed description of the algorithm.
_____________________________________

Download the code by cloning the git repository using

$ git clone https://github.com/rajeshmondal18/DviSukta
_____________________________________

The code starts by reading the real space gridded data and performing a 3D Fourier transform of it. Alternatively, it starts by reading the data already in Fourier space.

If your data is in real space, you need to install FFTW-3.x.x with the following flags: '--enable-float', '--enable-threads' and '--enable-openmp' to compile this set of codes. Look at the installation instruction http://www.fftw.org/fftw3_doc/Installation-on-Unix.html#Installation-on-Unix
_____________________________________

Use the makefile for compilation in the following manner:

$ make


It will create the executable 'bispec'.
_____________________________________

You need to specify 
1. Grid spacing in Mpc 
2. The number of k1 bins 
3. The number of n bins 
4. The number of cos(theta) bins 

in the input file.

An input data "c_data8.0_100" (binary data) and an input file "input.bispec" have been provided for testing.  
_____________________________________

To run:

$ ./bispec
_____________________________________

Please acknowledge Mondal et al. 2021 (https://arxiv.org/abs/2107.02668) if you are using this code.
