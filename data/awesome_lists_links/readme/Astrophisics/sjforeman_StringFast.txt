# StringFast

This code allows the user to calculate TT, EE, and BB power spectra (as separate scalar [for TT and EE], vector, and tensor contributions) for "wiggly" cosmic strings described by the Unconnected Segment Model from the following paper: 

> L. Pogosian & T. Vachaspati, Phys. Rev. D60, 083504 (1999) [(astro-ph/9903361)](https://arxiv.org/abs/astro-ph/9903361).

The code interpolates between a suite of precomputed outputs of [`CMBACT`](http://www.sfu.ca/~levon/cmbact.html), the public code that accompanies that paper. The properties of the strings are described by four parameters:

- Gmu: dimensionless string tension
- nu: rms transverse velocity (as fraction of c)
- alpha: "wiggliness"
- xi: comoving correlation length of the string network

It is written as a Fortran 90 module, and is described in further detail in the following paper:

> "Predicted Constraints on Cosmic String Tension from Planck and Future CMB Polarization Measurements" by S. Foreman, A. Moss, & D. Scott, Phys. Rev. D84, 043522 (2011) [(astro-ph/1106.4018)](https://arxiv.org/abs/1106.4018).

See the header of the source file for usage instructions. 

## Current version: 1.0 (June 2011)

The tarball contains:

- `stringfast.f90`, the main source file;
- 8 data files with suffix `.sfp`, which are read into memory by the module at runtime and must remain in the same directory as the executable file;
- `driver.f90`, a small driver program illustrating a simple use of the module;
- and a Makefile for the driver program. 
