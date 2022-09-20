TACHE - TensoriAl Classification of Hydrodynamic Elements 
===========================================================

[![DOI](https://zenodo.org/badge/109244123.svg)](https://zenodo.org/badge/latestdoi/109244123)

This Fortran code performs tensor classification on finite element hydrodynamics simulations - currently smoothed particle hydrodynamics (SPH) simulations only

Science using these algorithms was first published in Forgan et al (2016), Monthly Notices of the Royal Astronomical Society, Volume 457, Issue 3, p.2501-2513, DOI: 10.1093/mnras/stw103

HEALTH WARNING: This is a heavily refactored combination of several codes used in the above work, and as such is still in testing

Features:
--------
* Reads in SPH snapshot files (currently sphNG formats only)
* Computes neighbour lists for SPH data (assuming snapshot's smoothing lengths)
* Computes either the (symmetric) velocity shear tensor or tidal tensor, and their eigenvalues/eigenvectors
* Classifies fluid elements by number of "positive" eigenvalues
* Permits decomposition of snapshots into classified components
* Python plotting scripts

Features in Development:
-----------------------
* Spiral Fitting Algorithms (Forgan et al, in review)


Future Features/Wishlist:
-------------------------
* Ability to read in more SPH file formats
* Ability to process grid, Voronoi and meshless simulations

Requirements:
-------------
* Fortran compiler (gfortran recommended) and Makefile software (e.g. gmake)
* Python for plotting scripts (scripts developed in Python 2.7) - dependencies include numpy, matplotlib and f2py


Execution:
----------
To compile the code, navigate to the src/ directly and type

`> make`

to compile the main program

Once compiled, the code is executed with the command

`> ./tache`

The code reads in a single input parameter file `tache.params`, which should be modified before execution

The accompanying `spiralfind` program is compiled via

`>make spiralfind`

and run by

`>./spiralfind`

Which reads in `spiralfind.params` upon execution.

Example parameter files for both `tache` and `spiralfind` are available in the `paramfiles` directory
