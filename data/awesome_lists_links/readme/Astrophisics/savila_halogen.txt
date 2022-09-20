=======
HALOGEN
=======

HALOGEN is a C program for creating realistic synthetic dark matter halo catalogs 
in a flash. 

It decomposes the problem of generating cosmological tracer distributions
(eg. halos) into four steps:
* Generating an approximate density field;
* Generating the required number of tracers from a CDF over mass;
* Placing the tracers on field particles according to a bias scheme dependent
on local density;
* Assigning velocities to the tracers based on velocities of local particles.

It also implements a default set of four models for these steps as:
* 2LPT density field
* Tabulated mass function from the literature
* A single-parameter power-law biasing scheme.
* A linear transformation of velocities from the sampled DM particle
velocities.

For more details on the method, see Avila, S. et
al. 2015 -- http://mnras.oxfordjournals.org/cgi/content/abstract/stv711?ijkey=Do3EfS9S1jHUhrv&keytype=ref

Installation
------------
HALOGEN depends only on the libraries FFTW (v2.x only) and GSL. To set these and 
other specific options for compilation, edit the top part of the ``Makefile.defs``
file in the top-level directory. Compilation has been tested with mpicc/gcc, 
icc and craycc.

Also in the ``Makefile.defs`` file are a number of global options for HALOGEN
which affect its output in various ways. Please set these as desired.

After modifying ``Makefile.defs``, all (4) executables can be compiled with
``make`` (using ``make halogen``, ``make 2LPT-HALOGEN``, ``make 2LPT`` and
``make fit`` makes each executable individually).

The executables will be compiled into the top-level directory.

To optionally install, save the executable somewhere on the system path, eg. 
in ``/usr/bin/``.


Usage
-----
The primary usage is by using the fully combined executable, and simply runs
as

    $ 2LPT-HALOGEN <2LPT.input> <HALOGEN.input>

For details on the parameters in the input files, and how to run the other 
executables, see the README file in the ``examples/`` directory.

We strongly encourage every user to run through these examples before
undertaking any other usage of HALOGEN.


Acknowledgments
---------------
If you find this code helpful in your research, please cite Avila, S. et
al. 2014 -- http://arxiv.org/abs/1412.5228.

Note that HALOGEN makes use of the pre-existing codes:
 - 2LPTic by Sebastian Pueblas and Roman Scoccimarro (http://arxiv.org/abs/astro-ph/9711187, http://cosmo.nyu.edu/roman/2LPT/)
 - CUTE  by David Alonso (http://arxiv.org/abs/1210.1833)

Authors
-------
Santiago Avila Perez: santiago.avila@uam.es
Steven Murray: steven.murray@uwa.edu.au 
