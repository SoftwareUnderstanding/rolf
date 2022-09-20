# aesop

[![Build Status](https://travis-ci.org/bmorris3/aesop.svg?branch=master)](https://travis-ci.org/bmorris3/aesop) [![Documentation Status](https://readthedocs.org/projects/arces/badge/?version=latest)](http://arces.readthedocs.io/en/latest/?badge=latest) [![Powered by Astropy Badge](http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat)](http://www.astropy.org) 


ARC Echelle Spectroscopic Observation Pipeline (aesop)


The ARC Echelle Spectroscopic Observation Pipeline, or ``aesop``, is a high resolution 
spectroscopy software toolkit tailored for observations from the Astrophysics Research 
Consortium (ARC) Echelle Spectrograph mounted on the ARC 3.5 m Telescope at Apache 
Point Observatory. ``aesop`` picks up where the traditional IRAF reduction scripts leave 
off, offering an open development, object-oriented Pythonic analysis framework for echelle
spectra. 

Basic functionality of ``aesop`` includes: (1) blaze function normalization by polynomial 
fits to observations of early-type stars, (2) an additional/alternative robust least-squares 
normalization method, (3) radial velocity measurements (or offset removals) via 
cross-correlation with model spectra, including barycentric radial velocity calculations, 
(4) concatenation of multiple echelle orders into a simple 1D spectrum, and (5) approximate
flux calibration. 

For more info, [read the docs](http://arces.readthedocs.io/en/latest/?badge=latest)!