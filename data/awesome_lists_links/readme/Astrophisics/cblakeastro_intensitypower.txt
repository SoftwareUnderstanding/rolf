# intensitypower
This package is a set of python functions to measure and model the auto- 
and cross-power spectrum multipoles of galaxy catalogues and radio 
intensity maps presented in spherical co-ordinates.  Code is also 
provided for converting the multipoles to power spectrum wedges P(k,mu) 
and 2D power spectra P(k_perp,k_par).

This package accompanies the paper "Power spectrum modelling of galaxy 
and radio intensity maps including observational effects".

https://arxiv.org/abs/1902.07439

We assume that the galaxy catalogue is a set of discrete points, and the 
radio intensity map is a pixelized continuous field which includes:

* angular pixelization using healpix
* binning in redshift channels
* smoothing by a Gaussian telescope beam
* addition of a Gaussian noise in each cell

The galaxy catalogue and radio intensity map are transferred onto an FFT 
grid, and power spectrum multipoles are measured including curved-sky 
effects.  Both maps include redshift-space distortions.

The package includes wrappers for two example applications -- datasets 
built from a 30x30 deg cone cut from the GiggleZ N-body dark matter 
catalogue (http://www.astronomy.swin.edu.au/~gpoole/GiggleZ.html), and 
datasets generated from galaxy catalogues presented by the MICE 
simulation (http://maia.ice.cat/mice) for 0 < RA < 90, 30 < Dec < 90.

The python codes presented are:

* rungiggpk.py -- build mock galaxy and intensity mapping datasets from 
the GiggleZ dark matter simulation, and measure and model their auto- 
and cross-power spectrum multipoles.

* runmicepk.py -- measure and model the auto- and cross-power spectrum 
multipoles of galaxy catalogues and intensity mapping datasets generated 
from the MICE simulation.

* hppixtogrid.py -- transfer (redshift, healpix) binning to (x,y,z) 
binning using Monte Carlo random catalogues.

* getcorr.py -- generate corrections to the power spectra for the 
various observational effects.

* measpk.py -- model and measure the auto- and cross-power spectrum 
multipoles of the galaxy catalogue and density field.

* pktools.py -- set of functions for modelling and measuring the power 
spectra.

* boxtools.py -- set of functions for manipulating the survey cone 
within the Fourier cuboid.

* sphertools.py -- set of functions for manipulating the (redshift, 
healpix) density field.

* micetools.py -- set of functions for reading in and applying FASTICA 
foreground subtraction to the MICE simulations.

The python libraries needed to run the functions are:

* numpy

* scipy

* healpy

* astropy

* matplotlib

* numpy_indexed

* sklearn

The other accompanying files are:

* GiggleZ_z0pt000_dark_subsample.ascii.gz -- dark matter subsample of the 
GiggleZ N-body simulation, used to produce the example mock datasets.

* winx_MICEv2-ELGs.dat.gz -- window function for the MICE ELG sample 
(raw intensity and galaxy data files are large and available on request).

* pkcambhalofit_zeq0_gigglez.dat -- CAMB halofit model power spectrum 
with the same fiducial cosmology as GiggleZ at z=0.

* pkcambhalofit_zeq0pt4_mice.dat -- CAMB halofit model power spectrum 
with the same fiducial cosmology as MICE at z=0.4.

* pixwin_nside128.dat -- healpix window function for nside=128 (extended 
to higher multipoles than provided by healpix).

* pkpole_rungiggpk.png -- plot of power spectrum multipole model and 
measurements for the default GiggleZ simulation analysis.

* pkpole_runmicepk.png -- plot of power spectrum multipole model and 
measurements for the default MICE simulation analysis.

* pkwedge_runmicepk.png -- same for power spectrum wedges

* pk2d_runmicepk.png -- same for 2D power spectra
