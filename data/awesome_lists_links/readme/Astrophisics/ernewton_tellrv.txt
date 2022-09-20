# Absolute near-infared radial velocities

This code measures absolute radial velocities for low-resolution NIR spectra. I use telluric features to provide absolute wavelength calibration, and then cross-correlate with a standard star. To use the code for your star, you will need observations of a standard star (included) and your science target (examples included). You will need both the telluric and non-telluric-corrected spectra and the FITS headers, but no other spectra is required. Orders do not need to be combined. It should work for all extant spectra taken with SpeX on IRTF or similar instruments. 

A radial velocity standard is included in this installation (`/spec/J0727+0513_rest.fits`). It was created with the program `makemeastandwich.pro`. The wavelength calibration has been performed, but the known RV (18.2 km/s) and the barycentric velocity have not been corrected for. These are taken into account in nir_rv.

## Routines

### check_standards
Calculates radial velocities of RV standards. Use to test functionality and as an example of usage. Please check this with the cross-correlation routine you are using! The measured RVs should be within 6 km/s of the recorded value.

### nir_rv
Calculate the radial velocity from a NIR spectrum. Performs both the wavelength calibration and the cross-correlation with a standard star, and takes into account the barycentric velocities and the RV of the standard star.

### ern_rv
Cross-correlate the science target with the standard star to measure relative radial velocity. See below for notes on the cross-correlation subroutines and flattening subroutines that can be used.

### order_variables
Get required information from the FITS header and get pre-selected wavelength regions to provide to other routines. If this information is not provided, it will be estimated from the spectrum.

## Atmospheric transmission spectrum

This code requires an atmospheric transmission spectra. I use Lord (1992) ATRANS. This is available in the standard Spextool library and by default the code will look for it there. If you do not have ATRANS, you can download it from: [GEMINI](http://www.gemini.edu/sciops/telescopes-and-sites/observing-condition-constraints/ir-transmission-spectra)
[SOFIA](https://atran.sofia.usra.edu/cgi-bin/atran/atran.cgi)

## Alternative sub-routines

### Cross-correlation
There are three options for cross-correlation routines: `xcorl`, `c_correlate`, and `cross_correlate`. These are used in `ern_rv`, and can be selected by keyword in the top-level routine `nir_rv`. c_correlate is the default since I believe it is the most commonly available, and ought to be included exist in any modern IDL distribution. `xcorl` is not my code, but may be available to you. I prefer the routine `cross_correlate` which may be available in your installation. `xcorl` and `cross_correlate` produce consistent results; I have improved the peak finding after retrieving results from `c_correlate` but values may still differ by up to 1 km/s.

Note Jun 29 2018: `c_correlate` is producing discrepant behavior due to spikes in the cross-correlation function occurring at 0. This has been rectified by testing lag=0.01 instead of lag=0. but I still do not understand the origin of this error.

Note Nov 30 2018: Aaron Rizzuto identified this as being caused by the last item in the flux array being identically zero after flattening. Forcing them to be the median of the flattened array fixes this problem. The lag=0.01 hack removed and c_correlate works as expected.

### Continuum fitting
There are two options for continuum fitting: a spline-based routine and the routine `contf`. `contf` is not my code, but may be available to you. The default spline-based routine is based on `contf` and is included in this code. The choice is not important in most cases.

## Reference

If you use this code in published research, please cite Newton et al. (2014): http://adslabs.org/adsabs/abs/2014AJ....147...20N/

[![DOI](https://zenodo.org/badge/4705/ernewton/tellrv.svg)](https://zenodo.org/badge/latestdoi/4705/ernewton/tellrv)

## License

Copyright 2015 Elisabeth R. Newton. Licensed under the terms of the MIT License (see LICENSE).
