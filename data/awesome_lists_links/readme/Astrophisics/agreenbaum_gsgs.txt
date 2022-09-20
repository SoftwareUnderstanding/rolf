# gsgs - A Greenbaum-Sivaramakrishnan style Gerchberg-Saxton
GSGS depends on some functions and utilities in the poppy package:

Matrix_dft, zernike available at [https://github.com/mperrin/poppy](https://github.com/mperrin/poppy). 

It's also a good idea to have astropy, but here I only use it for fits files.
## What is available in this package:

The core program here is gsalgo.py - this computes all the Fourier transforms, etc.

Provided are also:

* zerniketools.py - for zernike smoothing an simulating aberrations. Put together mostly using a few programs from Tim van Werkhoven's [python](https://github.com/tvwerkhoven)
* simtools.py - some stuff for generating fake data, pupils, etc.

###GSGS Demo###

I've written a short example driver, test_gs.py. You should be able to run it right from the command line:

	python test_gs.py

The demo procedure is essentially contained in the simple_demo function. Various functions called have "debug" options that you can turn on to get a better idea of what that step is doing. 

A short explanation of the procedure

* Set optical parameters - wavelength, pixel scale, telescope diameter, etc.
* Generate an aberration given some chosen set of Zernike functions and user-defined P-V value
* Create NRM and full pupil PSFs using this aberration
* Measure mask hole pistons from the NRM image and use it to create a pupil estimate array
* Run gsgs using your PSF, known pupil, and pupil estimate

