# ReionYuga
This is an open-source code to generate the Epoch of Reionization (EoR) neutral Hydrogen (HI) field (successively the redshifted 21-cm signal) within a cosmological simulation box. It is based on semi-numerical techniques. It is written in C. The code is coupled with our N-body code https://github.com/rajeshmondal18/N-body and friends-of-friends (FoF) Halo finder https://github.com/rajeshmondal18/FoF-Halo-finder.
_____________________________________

Read the user's guide 'ionz_flow_chart.pdf' to understand the algorithm.
_____________________________________

Download the code by cloning the git repository using

$ git clone https://github.com/rajeshmondal18/ReionYuga

_____________________________________

You need to install FFTW-3.x.x with the following flags: '--enable-float', '--enable-threads' and '--enable-openmp' to compile this set of codes. Look at the installation instruction http://www.fftw.org/fftw3_doc/Installation-on-Unix.html#Installation-on-Unix

_____________________________________

Use the makefile for compilation in the following manner:

$ make ionz_main

It will create the executable 'ionz_main'. To run

$ ./ionz_main
_____________________________________

Please acknowledge Choudhury et al. 2009 (http://adsabs.harvard.edu/abs/2009MNRAS.394..960C), Majumdar et al. 2014 (http://adsabs.harvard.edu/abs/2014MNRAS.443.2843M) and Mondal et al. 2017 (http://adsabs.harvard.edu/abs/2017MNRAS.464.2992M), if you are using this code.
