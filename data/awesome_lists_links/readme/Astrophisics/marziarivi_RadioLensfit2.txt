# RadioLensfit2

**Radio Weak Lensing shear measurement in the Visibility Domain**

This is a complete version of [RadioLensfit](https://github.com/marziarivi/RadioLensfit).
Three tools are provided:

**Simulate**

Simulation of a radio weak lensing observation with a given reduced cosmic shear and a given source catalog.
The source catalog is text file containing for each source (row) l,m position (rad), flux (uJy), fwhm (arcsec), ellipticity components e1,e2.
Radio telescope configuration and observing time sampling must be provided in a Measurement Set.
The effect of the input reduced cosmic shear is applied to galaxies ellipticity. 
The corresponding simulated visibilities observed by the radio telescope are written in the DATA column of the same Measurement Set (I stokes component). 
The instrument gaussian noise variance is computed based on the antenna SEFD of the radiotelescope and time and frequency sampling interval.

Usage: *Simulate* (filename MS) (source catalog filename) (number of sources) (shear coord_1) (shear coord_2)
 
MPI version is also available.

**RadioLensfit2**

Measurement of star forming galaxy ellipticies from a radio weak lensing observation.
Data visibilities and observation configuration must be provided in a Measurement Set. If the MS contains 4 polarizations, I stokes is computed from them, otherwise a single polarization is expected to contain I stokes.
The number of galaxies and the corresponding source catalog (ordered by decreasing flux or SNR) containing at least source SNR, position and flux must be provided (scalelength is optional but highly recommended). Source position and flux (and scalelength) are used for source visibilities extraction, then shape measurement is performed according to RadioLensfit methodology: a single model fitting approach where the likelihood is marginalised over position, flux and scalelength source parameters.  
The list of galaxies with the measured ellipticities is written in a text file called *ellipticities.txt*.

Serial version usage: *RadioLensfit2* (source catalog filename) (number of sources) (filename MS)

MPI version usage: *RadioLensfit2-mpi* (source catalog filename) (number of sources) (filename MSs prefix)

For the MPI versions, MS must be split in individual spectral windows. All MS must have the same name ending with the number of the spectral window and extension  ".MS".
Filename prefix consists in the common part of all the MS name, i.e. (without the final number and extension)  
 
**shear.py** 

Shear measurement from weighted average of the measured ellipticity of the galaxies in the field of view. 
To be modified...


# Installation

Requirements:
- CASACORE library for Measurement Set I/O
- GSL library 
- MPI library (for the parallel version)

Edit the Makefile:

- enable/disable OpenMP (default: enabled)
- enable/disable MPI (default: enabled)
- default source shape model is exponential, other models allowed are GAUSSIAN or MATCH_EXP (Galsim Exponential matched Gaussian FWHM)   

- update CASACORE include and library path
- set the compiler and compilation flags you want to use (default: GNU)

Check/change default parameters in the *default_params.h* file 

*make all*

# Citing RadioLensfit2

If you use RadioLensfit2 and find it useful, please consider citing the related paper:

Rivi M., Miller L., *RadioLensfit: an HPC Tool for Accurate Galaxy Shape Measurement with SKA*, 2022, Astronomy & Computing,39,100574 - [arXiv:2203.14071](https://arxiv.org/abs/2203.14071)

Astrophysics Source Code Library ID ascl:2208.019 (https://ascl.net/2208.019.)



