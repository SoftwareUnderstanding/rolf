[![DOI](https://zenodo.org/badge/336397843.svg)](https://zenodo.org/badge/latestdoi/336397843)

## OVERVIEW

This software package was originally used to fit the Spitzer
mid-infrared spectra of the QUEST (**Q**uasar **U**LIRG and
**E**volution **ST**udy) sample, as described in [Schweitzer et
al. 2006](https://ui.adsabs.harvard.edu/abs/2006ApJ...649...79S/abstract),
[Schweitzer et
al. 2008](https://ui.adsabs.harvard.edu/abs/2008ApJ...679..101S/abstract),
and [Veilleux et
al. 2009](https://ui.adsabs.harvard.edu/abs/2009ApJS..182..628V/abstract). It
uses two PAH templates from [Smith et
al. 2007](https://ui.adsabs.harvard.edu/abs/2007ApJ...656..770S/abstract)
atop an extincted and absorbed continuum model to fit the mid-IR
spectra of galaxies that are heavily-absorbed and AGN with silicate
emission.

The current version of `QUESTFIT` is optimized for processing spectra
from the CASSIS (**C**ombined **A**tlas of **S**ources with
**S**pitzer **I**RS **S**pectra)
[portal](https://cassis.sirtf.com/atlas/welcome.shtml) to produce PAH
fluxes for heavily absorbed sources. This method is described in Spoon
et al. 2021 (submitted). These PAH fluxes will appear in the IDEOS
(**I**nfrared **D**atabase of **E**xtragalactic **O**bservables from
**S**pitzer) [portal](http://ideos.astro.cornell.edu/).

## REQUIREMENTS

IDL (tested with v8.5; may work with pre-v8.0 versions)

IDL libraries:
- [IDL Astronomy User's Library](http://idlastro.gsfc.nasa.gov)
- [Coyote](http://www.idlcoyote.com/documents/programs.php#COYOTE_LIBRARY_DOWNLOAD), for graphics AND undefine.pro
  - or from the [GitHub repository](https://github.com/davidwfanning/idl-coyote/tree/master/coyote)
- [ISAP](https://old.ipac.caltech.edu/iso/isap/isap.html)

Notes:
- The Spitzer IRS data reduction software SMART ships with a slightly
modified version of the ISAP library. At least one instance of errors
has been documented using this version of the library.
- The IDL Astronomy User's Library ships with some Coyote
routines. However, it's not clear how well these libraries keep track
of each other, so it may be preferable to download each package
separately and delete the redundant routines that ship within other
packages.

## QUICK TEST

See files in the `questfit/test` subdirectory:

IDEOS ID 4978688_0 = Mrk 231

IDL> questfit, control file='4978688_0.ideos.cf',pathin='[path-to-questfit]/questfit/test/',pathout='[output-directory]',/res,/ps,/data,/log

IRAS21219

IDL> questfit, control file='IRAS21219m1757_dlw_qst.cf',pathin='[path-to-questfit]/questfit/test/',pathout='[output-directory]',/res,/ps,/data,/log

## DETAILED USAGE

`questfit` and `ideos_readcf` are used to peform the spectral
fitting. The other `ideos_*` routines are for collating the PAH fluxes
for input to the IDEOS database.

The control file (*.cf) consists of 10 space-separated text columns of any width:

| A          	| B            	| C   	| D   	| E        	| F    	| G  	| H 	| I    	| J   	|
|------------	|--------------	|-----	|-----	|----------	|------	|----	|---	|------	|-----	|
| source     	| spectrum.xdr 	| -1  	| -1. 	| dummy    	| 0.0  	| 0. 	| X 	| 0.0  	| 0.0 	|
| template   	| si1.xdr      	| 0.1 	| 0.  	| DRAINE03 	| 10.0 	| 0. 	| S 	| 0.0  	| 0.0 	|
| absorption 	| tau1.xdr     	| 1.0 	| 0.  	| H2Oice6  	| 0.0  	| 0. 	| S 	| 0.0  	| 0.0 	|
| blackbody  	| BB           	| 0.1 	| 0.  	| DRAINE03 	| 1.0  	| 0. 	| S 	| 100. 	| 0.0 	|
| absorption 	| tau1.xdr     	| 0.0 	| 1.  	| H2Oice6  	| 1.0  	| 0. 	| S 	| 0.0  	| 0.0 	|
| powerlaw   	| PL           	| 0.1 	| 0.  	| DRAINE03 	| 1.0  	| 0. 	| S 	| -3.4 	| 0.0 	|
| absorption 	| tau1.xdr     	| 1.0 	| 0.  	| H2Oice3  	| 1.0  	| 0. 	| S 	| 0.0  	| 0.0 	|
| absorption 	| tau2.xdr     	| 1.0 	| 0.  	| H2Oice6  	| 1.0  	| 0. 	| S 	| 0.0  	| 0.0 	|
| extinction 	| draine03.xdr 	| 0.0 	| 0.  	| DRAINE03 	| 0.0  	| 0. 	| X 	| 0.0  	| 0.0 	|

Meaning of the columns:

- A
  - The type of data. Include at least one of each datatype
    (`template, BB, PL, absorption, extinction`). You have to use for
    at least one of each datatype (`template, BB, PL`) an absorption
    (as in the file above). And at least one extinction file has to be
    given. If you want to use more then one absorption on a certain
    datatype just add them up under the regarding datatype and they
    will all work on the datatype above (like for the powerlaw in the
    example above).

- B
  - `source, template, absorption, extinction`: filename
  - 'BB, PL`: any string

- C
  - `source`: lower wavelength limit. "-1" will use the lowest possible
    common wavelength.
  - `template, blackbody, powerlaw`: normalization factor
  - `absorption`: tau_peak
  - `extinction`: any float

- D
  - `source`: upper wavelength limit. "-1" will use the largest
    possible common wavelength.
  - `template, blackbody, powerlaw`: fix/free parameter for the
    normalization. 1=fixed, 0=free 
  - `absorption`:  fix/free parameter for tau_peak. 1=fixed, 0=free 
  - `extinction`: any float

- E
  - `source, absorption`: any string
  - `template, blackbody, powerlaw`: shorthand for the extinction curve
  - `extinction`: shorthand for the extinction curve is defined
    and connected to the xdr.file

- F
  - `source, extinction, absorption`: any float
  - `template, blackbody, powerlaw`: extinction value (A_V)
 
- G
  - `source, extinction, absorption`: any float
  - `template, bl, powerlaw`: fix/free parameter for A_V. 1=fixed,
    0=free

- H
  - `source, extinction, absorption`: any string
  - `template, blackbody, powerlaw`: S=screen extinction, M=mixed extinction

- I
  - `source, template, absorption, extinction`: any float
  - `blackbody`: temperature (in K)
  - `powerlaw`: index

- J
  - `source, template, absorption, extinction`: any float
  - `blackbody`: fix/free parameter for temperature. 1=fixed, 0=free
  - `powerlaw`: fix/free parameter for powerlaw index. 1=fixed, 0=free

To run:

*IDL> questfit,control file='control.cf'*

The fit will start and produce a plot with the model spectra fitted to
the source. In addition the contribution of each model depending on
wavelength is plotted. This is not in multicolor. If you want to have
a nicer view, create a eps file by typing :

*IDL> questfit,control file='control.cf',ps=1*

This creates an eps-file into the 'pathout' directory At the same time
the fitoutput which appears in your shell (after the fit) will be
written into a file if you use ps=1.

The filename for the eps file is *fit:sourcename.xdr_par:control
file.cf.eps*.

The filename for the data file is
*fitresult:sourcename.xdr_par:control file.cf.dat*.

In the eps plot each data type (`template, BB, PL`) has a common
color.  Within one type the spectra differ from each other by the
linestyle.

At the same time on the command shell at the end of the fit some
information appears. The output is translated as follows:

- Each value P(X) is the program internal definition for the fitting
  parameter.
- The numbers of the contributors describe the order of appearance in
  the control file.
- `norm factor` is total normalization factor/local maximum.  This is
  the factor you have to put fixed into the control file if you like
  the result. Be careful if you fix the parameters and you change the
  wavelength range, as the norm factor depends on the maximum found in
  the wavlength range.
- `total normalization factor` is the value which you have to multiply
  to the original template to get the fitted result.
- `integral [W/cm^2] of XX` is the flux integrated over frequency.

- `contribution [%] to integrated source flux` is the contribution to
  the integrated flux of each contributor

- `Sum of all contributions [%] relative to int. sourceflux` adds up
  all contributions; should be 100% in the case of a perfect fit.

Additional options:

- *IDL> questfit,control file='control.cf',res=1*: shows the residual
   (Data/Model)

- *IDL> questfit,control file='control.cf',log=1*: uses log xy-axis

- *IDL> questfit,control file='control.cf',data=1* will produce
  output-data files of each model with norm-factors not equal to
  0. The content of this output file is wavelength in microns, flux in
  Jy.The filenames are for example:
  - *sourcename.xdr_par:controlfile.cf_template1.fit*
  - *sourcename.xdr_par:controlfile.cf_BB2.fit*
  - *sourcename.xdr_par:controlfile.cf_total.fit*

- *IDL> questfit,controlfile='control.cf',ident=1*: shows an
  identification index for the spectra

## QUESTIONS? BUGS? WANT TO MODIFY THE CODE?

Feel free to contact David Rupke at drupke@gmail.com with questions,
bug reports, etc.

Modifications are encouraged, but subject to the license.

## LICENSE AND COPYRIGHT

Copyright (C) 2016--2021 David S. N. Rupke, Vince Viola, Henrik Sturm,
Mario Schweitzer, Dieter Lutz, Eckhard Sturm, Dong-Chan Kim, Sylvain
Veilleux

These programs are free software: you can redistribute them and/or
modify them under the terms of the GNU General Public License as
published by the Free Software Foundation, either version 3 of the
License or any later version.

These programs are distributed in the hope that they will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with these programs.  If not, see http://www.gnu.org/licenses/.
