# ZeldovichRecon
Code to compute the post-reconstruction halo correlation function using the Zeldovich approximation.

This code closely follows the formalism described in
`Reconstruction within the Zeldovich approximation`
MNRAS 450(2015)3822, [arxiv:1504.03677](https://arxiv.org/abs/1504.03677)
but has been updated to include the extensions described in
`The reconstructed power spectrum in the Zeldovich approximation`
(by Chen, Vlah & White 2019).  This code has been checked against
the Python code from that paper, which is available at
https://github.com/sfschen/ZeldovichReconPk
and produces both P(k) and xi(r).

There are several variants here.  First, a "stand-alone" C++ code,
zelrecon.cpp, which computes the various contributions to the correlation
function.  This should compile under g++.

There is a version, zelrecon_ctypes.cpp, which is designed to be called
from Python using the ctypes library.  The wrapper "zelr.py" gives an
example [you need to edit it to set paths correctly].  When compiling
the code under g++ I use -O -funroll-loops -fopenmp -shared -fPIC.  This
code is not being kept up-to-date and should be considered **deprecated**.

Finally, there is a version which implements the "ZEFT" formalism of
``A Lagrangian effective field theory'' [arxiv:1506.05264]
including the alpha term described in that paper.
This version is designed to be called from Python through the ctypes
library.  An example is given in zeftr.py and you should consider the
same compiler flags as above.  An example Makefile is included.
