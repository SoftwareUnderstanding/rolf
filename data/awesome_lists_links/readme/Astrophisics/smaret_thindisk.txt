Thindisk
========

`thindisk` is a simple Python program to compute the line emission
from a geometrically thin protoplanetary disk. It creates a datacube
in FITS format that can be processed in a data reduction package (such
as GILDAS) to produce synthetic images and visibilities. These
synthetic data can be compared with observations to determine the
properties (e.g. central mass or inclination) of an observed disk.

The disk is assumed to be in Keplerian rotation at a radius lower than
the centrifugal radius (which can be set to a large value, for a
purely Keplerian disk), and in infall with rotation beyond the
centrifugal radius.

Input file
----------

Model parameters are read from an input file. Here is an example:

```ini
[disk]
mstar = 0.20
incl = 85.
pa = 3.
r0 = 80
dist = 140
[cube]
npix = 1024
pixsize = 0.005
nchan = 128
chanwidth = 0.1
[line]
frequency = 219560.3541
intensity = gaussian, 135., 1.56
width = 0.1
vlsr = 6.0
[output]
name = l1527-c18o
```

Parameters
----------

- `mstar`:  mass of the central object in solar masses
- `incl`: disk inclination in degrees (90 for edge-on, 0 for
  face-on). Default: 45.
- `pa`: position angle of projected disk rotation axis, in degrees (0
  for a North-South, 90 for East-West). Default: 0.
- `rc`: centrifugal radius, in AU. Default: 1e4
- `size`: disk size, in AU. Default: infinite disk.
- `dist`: disk distance, in pc
- `npix`: number of pixels in RA and Dec offset. Default: 512.
- `pixsize`: size of a pixel, in arcsecs, Default: 0.1.
- `nchan`: number of velocity channels: Default: 128.
- `chanwidth`: channel width, in km/s. Default: 0.1.
- `frequency`: line frequency, in MHz
- `intensity`: line intensity of the disk surface. For a Gaussian
  distribution, set to `gaussian,int0,fwhm` where `int0` is the peak
  intensity, and `fwmh` is the FWHM (in arcsecs). For a power-law, set
  to `powerlaw,int_r1,r1,int_expn` where `int_r1` is the intensity at
  the radius `r1` (in arcsecs), and `int_expn` is the powerlaw
  exponent. For a ring, set to `ring,int_ring,r1,r2`, where `int_ring`
  is the intensity between radii `r1` and `r2` (in arcsecs). For a
  tapered power-law [(Andrew et al., Apj 700, 1502, 2009, Eq. 4)](https://doi.org/10.1088/0004-637X/700/2/1502), set
  to `tapered_powerlaw,int_r1,r1,int_expn`, where `int_r1` is a
  normalization factor, `r1` is a characteristic radius (in arcsecs)
  and `int_expn` is the power-law exponent.
- `width`: line width, in km/s. To set the linewidth to a fraction of
  the Keplerian velocity, use e.g. `0.1*vkep`. Default: 0.1
- `vlsr`: source systemic velocity in the LSR, in km/s, Default: 0.
- `name`: base name of the output FITS file, Default: output.fits.

Usage
-----

```
% python thindisk.py input.ini`
```

where `input.ini` is the name of the output file.

Citation
--------

If you use this code in a scientific publication, please cite it using
this DOI:

[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.592492.svg)](http://dx.doi.org/10.5281/zenodo.592492)
