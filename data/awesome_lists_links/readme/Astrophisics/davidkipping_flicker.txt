# flicker
Flicker calculates the mean stellar density of a star by inputting the flicker observed in a photometric time series. Written in Fortran90, its output may be used as an informative prior on stellar density when fitting transit light curves.

This code is based on the paper Kipping, Bastien, Stassun, Chaplin, Huber & Buchhave, "Flicker as a Tool for Characterizing Planets Through Asterodensity Profiling", 2015, ApJ, 785, L32 (http://adsabs.harvard.edu/abs/2014ApJ...785L..32K)

The produces deterministic estimates of the stellar density, but we recently updated the model to produce probabilistic estimates in Angus & Kipping, "Probabilistic Inference of Basic Stellar Parameters: Application to Flickering Stars", 2016, ApJ, 823, L9 (http://adsabs.harvard.edu/abs/2016ApJ...823L...9A).

** We generally recommend users use this updated version, available at https://github.com/RuthAngus/flicker **
