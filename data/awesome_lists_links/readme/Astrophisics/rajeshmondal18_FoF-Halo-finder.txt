# FoF-Halo-finder
This is an open-source code to identify the location and size of collapsed objects (halos) within a cosmological simulation box. It is based on the friends-of-friends (FoF) algorithm. It is written in C. The code is coupled with our N-body code https://github.com/rajeshmondal18/N-body.
_____________________________________

Read the user's guide 'FoF_doc.pdf' to understand the algorithm.
_____________________________________

Download the code by cloning the git repository using

$ git clone https://github.com/rajeshmondal18/FoF-Halo-finder.git
_____________________________________

You need to install FFTW-3.x.x with the following flags: '--enable-float', '--enable-threads' and '--enable-openmp' to compile this set of codes. Look at the installation instruction http://www.fftw.org/fftw3_doc/Installation-on-Unix.html#Installation-on-Unix

Use the makefile for compilation in the following manner:

$ make fof_main

It will create the executable 'fof_main'. To run

$ ./fof_main
_____________________________________

Please acknowledge Mondal et al. 2015 (http://adsabs.harvard.edu/abs/2015MNRAS.449L..41M), if you are using this code.
