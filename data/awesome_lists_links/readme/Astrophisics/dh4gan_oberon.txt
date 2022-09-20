OBERON - OBliquity and Energy balance Run on N body systems
===========================================================

[![DOI](https://zenodo.org/badge/24103/dh4gan/oberon.svg)](https://zenodo.org/badge/latestdoi/24103/dh4gan/oberon)
[![Build Status](https://travis-ci.com/dh4gan/oberon.svg?branch=master)](https://travis-ci.com/dh4gan/oberon)

This C++ code models the climate of Earthlike planets under the effects of an arbitrary number and arrangement of other bodies, such as stars, planets and moons.

Science using this code was first published in Forgan (2016), Monthly Notices of the Royal Astronomical Society, Volume 463, Issue 3, p.2768-2780, DOI: 10.1093/mnras/stw2098

Features:
--------
* Simple orbital setup routines or direct cartesian vector input positions for bodies
* 1D latitudinal energy balance (LEBM) climate modelling
* 4th Order Hermite N Body integration (shared variable timestep)
* Obliquity evolution taken from Laskar (1986a, A&A, 157, 59) and Laskar (1986b A&A, 164, 437)
* Ability to checkpoint/restart simulations (with a health warning at albedo transitions)
* Algorithms to accommodate ice sheet melting
* Carbonate Silicate Cycle Modelling (Williams and Kasting 1997, Haqq-Misra et al 2016)
* Library of Python 2.7 plotting scripts 
* Library of example parameter setups to run

Possible Future Features/Wishlist:
-------------------------

* individual timestepping
* More sophisticated spin evolution (cf Mercury-T)
* More flexibility in input planet atmospheres

Requirements:
-------------
* C++ compiler (g++ recommended) and Makefile software (e.g. gmake)
* Python for plotting scripts (scripts developed in Python 2.7) - dependencies include numpy, matplotlib, scipy

The code reads in a single input parameter file, which contains a set of global parameters for all bodies in the simulation, along with specific parameters for each
body included in the simulation. Parameter files can either specify the initial positions of all bodies, or the initial Keplerian orbits of all bodies.

Further details of the parameter file structure can be found in the userguide in `\docs`, and example parameter files are given in `\paramfiles`

Once compiled, the code is executed with the command

`> ./oberon input.params`

The code was originally developed using the eclipse CDT.  We now recommend using the Makefile in `\src` to compile with g++.
