# grapus - GRavitational instability PopUlation Synthesis
================================================================

This repository executes population synthesis modelling of self-gravitating disc fragmentation and tidal downsizing in protostellar discs.

The algorithms in this code are described in the following publications

*Forgan & Rice (2013), MNRAS, 432, pp 3168-3185* (v1.0)

*Forgan et al (2018), MNRAS, 474, pp 5036-5048* (v2.0, addition of N Body physics)

If you plan to use this code for your own research, or if you would like to contribute to this repository then please get in touch with a brief description of what you would like to do.  I will seek co-authorship for any subsequent publications.


Features
--------

This code reads in pre-run 1D viscous disc models of self-gravitating discs, computes where fragmentation will occur (and the initial fragment mass).  It then 
allows these fragment embryos to evolve under various forces, including:

* Quasistatic collapse of the embryo
* Growth and sedimentation of the dust inside the embryo
* The formation of solid cores
* Migration due to embryo-disc interactions
* Tidal disruption of the embryo
* (Optional) Gravitational interactions with neighbour embryos

Compiling and Running
---------------------

The code is written in FORTRAN 90 throughout. The supplied Makefile compiles the 
program using gfortran through the command

`> make`

And run using the command

`> ./grapus`

The input parameters are specified in `grapus.params` - an example file is given in `paramfiles/`

Plotting
--------

The output files can be plotted using Python scripts found in the
`plot/` directory

The scripts were developed in Python 2.7, and depend on numpy and matplotlib

License
-------

This code is licensed under the MIT license - see LICENSE.md
