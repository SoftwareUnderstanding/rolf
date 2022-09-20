[![DOI](https://zenodo.org/badge/63365551.svg)](https://zenodo.org/badge/latestdoi/63365551)

-------------------------------------------------------------------------
OVERVIEW
-------------------------------------------------------------------------

IFSFIT is a general-purpose library for fitting the continuum,
emission lines, and absorption lines in integral field spectra
(IFS). It uses PPXF (the Penalized Pixel-Fitting method developed by
Cappellari & Emsellem, 2004, PASP, 116, 138) to find the best fit
stellar continuum (using a user-defined library of stellar templates
and including additive polynomials), or optionally a user-defined
method to find the best fit continuum. It uses MPFIT to simultaneously
fit Gaussians to any number of emission lines and emission line
velocity components. It will also fit the NaI D feature using analytic
absorption and/or emission-line profiles, or a generic absorption-line
doublet or multiplet function (such as MgII 2796/3802).

-------------------------------------------------------------------------
REQUIREMENTS
-------------------------------------------------------------------------

IDL v8.0 or higher (last tested with v8.6)

IDL libraries:
- IDL Astronomy User's Library, for various routines
  http://idlastro.gsfc.nasa.gov
- MPFIT, for non-linear least-squares fitting
  http://www.physics.wisc.edu/~craigm/idl/idl.html
- Coyote, for graphics AND undefine.pro
  https://github.com/idl-coyote/coyote
- PPXF, for stellar continuum fitting
  http://www-astro.physics.ox.ac.uk/~mxc/idl/#ppxf
- IDLUTILS, for SSHIFTROTATE and b-spline routines
  https://www.sdss.org/dr15/software/idlutils/
- DRTOOLS, for multicore processing
  https://github.com/drupke/drtools

These libraries must be downloaded by the end user and installed in a
location that is in the IDL path.

Note that the IDL Astronomy User's Library ships with some Coyote
routines, and IDLUTILS ships with the IDL Astronomy User's Library and
MPFIT. However, it's not clear how well these libraries keep track of
each other, so it is preferable to download each package separately
and delete the redundant routines that ship within other packages.

To fit stellar continua, templates are required. E.g., the population
synthesis models from Gonzalez-Delgado et al. (2005, MNRAS, 357, 945)
are available at

http://www.iaa.csic.es/~rosa/research/synthesis/HRES/ESPS-HRES.html

The enclosed routine IFSF_GDTEMP can be used to convert these tables
into a form readable by IFSF.

IFSFIT also ships with the external routine CONSEC.PRO written by John
Johnson.

-------------------------------------------------------------------------
USAGE
-------------------------------------------------------------------------

Usage is in principle straigtforward. The continuum and emission-line
fitting is run from the command line as

> IFSF,'initialization_file' [,cols=[low,high],rows=[low,high], etc.]

and then the results are processed using

> IFSFA,'initialization_file' [,cols=[low,high],rows=[low,high], etc.]

The latter produces plots of the lines fit, a table of emission line
parameters and a table of fit results, and an IDL save file containing
a "data cube" of emission line parameters. If desired, it will also
normalize the continuum around NaI D 5890, 5896 and estimate various
parameters of this feature.

The initialization file defines an IDL function that returns an
initialization structure, INITDAT. The tags of this structure, and a
description of each one, are found in INITTAGS.txt. Several tags are
required, but most are optional.

The initialization function should also define two other structures,
INITMAPS and INITNAD, and also call these as keyword arguments. The
first of these, INITMAPS, controls how IFSF_MAKEMAPS processes the
output of IFSFA into parameter maps. This routine produces various
emission-line, continuum, and absorption-line maps. The possible tags
for INITMAPS are described in INITTAGS.txt. If IFSF_MAKEMAPS is not
used, INITMAPS can be set to an empty structure.

The second of these other initialization structures, INITNAD, controls
how the region around the NaI D feature is fit using IFSF_FITNAD. This
routine fits emission and absorption-line models to HeI 5876 and NaI D
5890, 5896 and produces plots of the fits and a "data cube" with fit
parameters. The routine also estimates errors using Monte Carlo
methods. The error estimation can be sped up using multi-core parallel
processing. The possible tags for INITNAD are described in
INITTAGS.txt. If IFSF_MAKEMAPS is not used, INITNAD can be set to an
empty structure.

There are a considerable number of knobs that can be turned to
optimize/customize the fits and customize the outputs. An example
initialization file is included (init/IFSF_F05189.pro).

The other user-modifiable initialization procedure that is required is
one that sets the initial emission-line parameter structure for input
into MPFIT. Included in this version is a procedure that is optimized
for the GMOS instrument (init/IFSF_GMOS.pro) and contains line
parameters pertaining to strong emission lines in the wavelength range
4000-7000 A. However, this code can be relatively easily adapated to
other instruments and to include parameters (like fixed line ratios)
for other emission lines. The code has in the past been used to
successfully fit data from many instruments: GMOS, LRIS, NIFS, OSIRIS,
and WiFeS.

-------------------------------------------------------------------------
COOKBOOK FOR QSO DEBLENDING
-------------------------------------------------------------------------

(0) Create nuclear template

    IDL> ifsf_makeqsotemplate,'nuclearspectrum.fits','nucleartemplate.xdr'

(1) Fit total spectra with quasar + exponential starlight model +
    emission lines.
(2) Calculate starlight-only spectra.

    In the initialization file, the REFIT tag under the ARGSCONTFIT
    tag should be commented out. The tag QSOXDR under ARGSCONTFIT
    should equal the path and filename of the template created in step
    (0). The HOST tag should be a structure with one tag, DAT_FITS
    that gives the path and filename of the output starlight-only
    file.

    IDL> ifsf,'ifsf_qsodeblend_example'
    IDL> ifsfa,'ifsf_qsodeblend_example'

    Output: starlight.fits

(3) Spatially integrate spectra

    IDL> ifsr_spaxsum,'starlight.fits','starlight_spatiallyintegrated.fits',[x1,y1,x2,y2]
    IDL> ifsr_spaxsum,'datacube.fits','totalspectrum.fits',[x1,y1,x2,y2]

(4) Fit spatially-integrated starlight spectrum with stellar
    templates. Stellar templates must be included with the
    STARTEMPFILE tag.

    IDL> ifsf,'ifsf_qsodeblend_example_host',/oned
    IDL> ifsfa,'ifsf_qsodeblend_example_host',/oned

(5) Fit spatially-integrated total spectrum with quasar + stellar
    templates + polynomial + emission lines.

    IDL> ifsf,'ifsf_qsodeblend_example_total',/oned
         ifsfa,'ifsf_qsodeblend_example_total',/oned

(6) Choose best-fit SPS model from steps (4) and (5) and sum over
    ages.

    Re-run IFSF/IFSFA on chosen model (host or total) and then run:

    IDL> ifsf_makestellartemplate,'/path/fitdir/galaxyshorthand_0001.xdr','/path/starlighttemplate.xdr'

(7) Fit total spectra with quasar + SPS starlight model + emission
    lines.
(8) Calculate starlight-only spectra.

    Uncomment REFIT and STARTEMPFILE tags in IFSF_QSODEBLEND_EXAMPLE,
    and change name of DAT_FITS file under HOST tag so that previous
    starlight spectra are not overwritten.

    IDL> ifsf,'ifsf_qsodeblend_example'
    	 ifsfa,'ifsf_qsodeblend_example'

(9) Spatially integrate spectra.

    IDL> ifsr_spaxsum,'/path/starlight_iter1.fits','/path/starlight_spatiallyintegrated_iter1.fits',[x1,y1,x2,y2]

(10) Fit spatially-integrated starlight spectrum.

    IDL> ifsf,'ifsf_qsodeblend_host',/oned
	 ifsfa,'ifsf_qsodeblend_host',/oned

(11) Compare SPS fits to starlight-only spectrum. Iterate from step 6
     if (significantly) different. Finish if same (within tolerance).


-------------------------------------------------------------------------
IMPORTANT NOTES
-------------------------------------------------------------------------

IFSF_MANYGAUSS, the routine that evaluates the emission line
Gaussians, assumes constant dispersion (in A/pix) and that there are
no "holes" in the spectrum (i.e., that each wavelength follows the
next by simply adding the dispersion).  If you have a variable
dispersion or gaps in your wavelength array, either modify
IFSF_MANYGAUSS to fit your needs or use IFSF_MANYGAUSS_SLOW (which, as
its name implies, is slower).

-------------------------------------------------------------------------
OPTIMIZING PERFORMANCE
-------------------------------------------------------------------------

The primary task, IFSF, can be run with the keyword NCORES to
parallelize the analysis among any number of cores. This will speed up
execution by a factor that is not exactly NCORES but is still large.

PPXF can be sped up (by some significant, but as-yet-unknown, factor)
by calling a pre-compiled Fortran version of BVLS. The entire IFSF
routine will be sped up by a smaller amount that depends on the data
quality, number of emission lines fit, number of stellar templates,
etc. This is accomplished by following the recipe below. (Tested with
IDL v8.5.1, gcc v6.2.0 from http://hpc.sourceforge.net/, on a Machine
running Mac OS X El Capitan and the MacPorts X11 tools.)

1) Download a Fortran version of BVLS here:

      https://people.sc.fsu.edu/~jburkardt/f_src/bvls/bvls.html

   or here:

      http://www.netlib.org/lawson-hanson/index.html

   In the former case, the BVLS subroutines can be split up from the
   single downloaded file using F90SPLIT:

      https://people.sc.fsu.edu/~jburkardt/c_src/f90split/f90split.html

   In the latter case, the files will have to be extracted from the
   source code more carefully.

2) Rename the subroutine BVLS as BVLS1, which is what the wrapper
   routine (below) will actually call.

3) Copy the contents of the ./bvls/ subdirectory in the IFSFIT
   distribution to the same directory as the BVLS source code.

   BVLS_WRAPF.F is an edited version of the example wrapper program
   (VECADD_WRAPF.F) found here:

      http://www.physics.usyd.edu.au/guides/idl/IDL_External.html

   BVLSF90.SH is a shell script that will compile and link together
   the BVLS source code and wrapper program, and is again an edited
   version of the example compilation script (BVLS.SH) found here:

      https://people.sc.fsu.edu/~jburkardt/f77_src/bvls/bvls.html

4) Edit the shell script with the location of the resulting library
   file, and then run the script:

      % chmod +x bvlsf90.sh
      % ./bvlsf90.sh

   Note that GFORTRAN must be callable from the command line. If
   desired, edit the shell script with your own Fortran compiler and
   options.

4) Edit PPXF by commenting out these lines:

      ; BVLS, A, B, bnd, soluz, ITMAX=15*s[2], IERR=ierr
      ; if ierr ne 0 then message, 'BVLS Error n. ' + strtrim(ierr,2)

   and replacing them with these:

      soluz = dblarr(s[2])
      rnorm = 0d
      nsetp = 0
      w = dblarr(s[2])
      index = lonarr(s[2])
      ierr = 0
      result = $
         CALL_EXTERNAL('/Location/of/library/libbvls.dylib', 'bvls_', $
                       s[1],s[2],A,B,bnd,soluz,rnorm,nsetp,w,index,ierr)
      if ierr ne 0 then message, 'BVLS Error n. ' + string(ierr,format='(I0)')

   Note that some of these steps may be somewhat
   architecture-dependent. E.g., the argument 'bvls_' is determined by
   running

      % nm libbvls.dylib

   which yields '_bvls_' as the subroutine name within the
   library. The leading '_' must be dropped.

-------------------------------------------------------------------------
EXTINCTION CORRECTIONS
-------------------------------------------------------------------------

To calculate gas E(B-V) in IFSF_MAKEMAPS, switch on the EBV tag in
INIT as in the IFSF_QSODEBLEND_EXAMPLE.PRO initialization file. This
calculates EBV for total flux (summed over components) and the flux on
a component-by-component basis. The TITLE tags are plot labels, and
selecting APPLY produces corrected fluxes, outputs them to a file, and
applies the extinction correction to the spatially-integrated total
fluxes and to Halpha line fluxes, masses, energies, and momenta
computed in any outflow calculation. It does not presently apply the
extinction correction to line ratios or to plots of line flux maps.

Presently the correction is only done using the E(B-V) calculated from
the total flux (summed over components). I.e., you can correct the
flux on a component-by-component basis but it uses the total (summed
over components) E(B-V). IFSFIT will output the corrected fluxes into
XDR files with suffixes '.emlflxcor_pp.xdr' and
'.emlflxcor_med.xdr'. The '_pp' version does a correction on a
per-pixel basis, and uses the median value if E(B-V) could not be
calculated. The ‘_med’ version uses the median E(B-V) (calculated as a
spatial median) from the entire map. The WINDSTR output from
IFSF_MAKEMAPS will also contain the spatially-integrated corrected
fluxes in the tag E_FLUX_TOT. This in turn contains the
extinction-corrected, spatially-integrated fluxes in the structure
tags COMP_UNEXT_PP, COMP_UNEXT_MED, TOT_UNEXT_PP, and TOT_UNEXT_MED.

IFSFIT can implement E(B-V) on the stellar continuum within PPXF; set
EBV_STAR = 1 in INIT. It will Monte Carlo the errors over NITER
iterations on stellar E(B-V) by setting MCERRORS = {niter:
YOUR_VALUE_HERE}.

-------------------------------------------------------------------------
QUESTIONS? BUGS? WANT TO MODIFY THE CODE?
-------------------------------------------------------------------------

Feel free to contact David Rupke at drupke@gmail.com with questions,
bug reports, etc.

Modifications are encouraged, but subject to the license.

-------------------------------------------------------------------------
LICENSE AND COPYRIGHT
-------------------------------------------------------------------------

Copyright (C) 2013--2021 David S. N. Rupke

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License or any
later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see http://www.gnu.org/licenses/.
