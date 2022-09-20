#segueSelect

##AUTHOR

Jo Bovy - bovy at ias dot edu

If you find this code useful in your research, please cite
[arXiv:1111.1724](http://arxiv.org/abs/1111.1724). Thanks!


##INSTALLATION

Standard python setup.py build/install


##DEPENDENCIES

This package requires [NumPy](http://numpy.scipy.org/), [Scipy] (http://www.scipy.org/), [Matplotlib] (http://matplotlib.sourceforge.net/), and [Pyfits](http://www.stsci.edu/resources/software_hardware/pyfits). To use coordinate transformations, [galpy](https://github.com/jobovy/galpy) is required.


##BASIC DOCUMENTATION

segueSelect is a Python package that implements the model for the
SDSS/SEGUE selection function described in Appendix A of
[arXiv:1111.1724](http://arxiv.org/abs/1111.1724). It automatically
determines the selection fraction---the fraction of stars with good
spectra---as a continuous function of apparent magnitude for each
plate. The selection function can be determined for any desired sample
cuts in signal-to-noise ratio, u-g, r-i, and E(B-V).

To get started, download the files at
http://astro.utoronto.ca/~bovy/segueSelect/, put them in some directory,
define an environment variable SEGUESELECTDIR that points to this
directory, and untar the segueplates.tar.gz file (tar xvzf
segueplates.tar.gz).


After installing the package (python setup.py install) you can use the
package as

	from segueSelect import segueSelect
	selectionFunction= segueSelect(sample='G',sn=15,select='all')

to get the selection function for the SEGUE G-star sample, using a
signal-to-noise ratio cut of 15, and selecting all stars in the
spectroscopic sample in the G-star color range 0.48 <= g-r <= 0.55
(as opposed to select='program', which just uses the stars that were
targeted as G stars).

The selection function is determined on the fly, so sample selection
can be adjusted if desired. Relevant options are

    ug= if True, cut on u-g, (default: False)
    	if list/array cut to ug[0] < u-g< ug[1]
    ri= if True, cut on r-i,  (default: False)
    	if list/array cut to ri[0] < r-i< ri[1]
    sn= if False, don't cut on SN, 
    	if number cut on SN > the number (default: 15)
    ebv= if True, cut on E(B-V), 
    	 if number cut on EBV < the number (default: 0.3)

The type of selection function can be set separately for 'bright'
plates and 'faint' plates. The default for both is 'tanhrcut', which
is the selection function described in Appendix A of
[arXiv:1111.1724](http://arxiv.org/abs/1111.1724), but other options
include:

    type_bright= or type_faint=
    
        'constant': constant for each plate up to the faint limit for
	the bright/faint sample (decent for bright plates, *bad* for
	faint plates that never reach as faint as the faint limit)

        'sharprcut': use a sharp cut rather than a hyperbolic tangent
	cut-off at the faint end of the apparent magnitude range

The recommended setting is 'tanhrcut' for both bright and faint plates.

For a full list of options, do

    ?segueSelect

Once the selection function is initialized it can be evaluated as

     plate=2964
     value= selectionFunction(plate,r=16.)

where value is then the fraction of stars in the photometric sample
with a SEGUE spectrum passing the sample cuts for that plate number
and that r-band apparent magnitude. 'r=' can be an array, for quicker
evaluation.


##ADVANCED FUNCTIONALITY

Please look at the source code (segueSelect/segueSelect.py) for an
overview of the advanced capabilities of this package. Some useful
functions are


    selectionFunction.check_consistency(plate)

which will calculate the KS probability that the spectropscopic sample
was drawn from the underlying photometric sample with the model
selection function.


    selectionFunction.plot(plate=plate)

plots the selection function of this plate.


    selectionFunction.plot_plate_rcdf(plate)

plots the cumulative distribution function in r-band apparent
magnitude of the spectroscopic sample (red) and the photometric
sample+selection-function-model for this plate (these should look
similar and are what are used to calculate the KS probability).


    read_gdwarfs(file=_GDWARFALLFILE,logg=False,ug=False,ri=False,sn=True,
                 ebv=True,nocoords=False)

which reads the G stars (*not* just the dwarfs!) and applies the
color, SN, and E(B-V) cuts (same format as above). If
[galpy](https://github.com/jobovy/galpy) is installed, velocities will
also be transformed into the Galactic coordinate frame (read the
source for details).


##K STARS

The code can also determine the selection function for SEGUE K
stars. However, the bright/faint boundary seems to move around for K
stars as the survey progressed (rather than stay constant at r=17.8
mag), so the default selection function fails to give a reasonable
selection function for many plates. The K star selection function is
still a work in progress, but determining the bright/faint boundary on
a plate-by-plate basis seems to work for most plates. This can be done
by using

    selectionFunction= segueSelect(sample='K',sn=15,select='program',indiv_brightlims=True)

Again, testing of this selection function has been very limited, so
use care when using the K stars. More details from some email:

    Iterating with Katie Schlesinger, I think I have mostly figured
    out the SEGUE K star selection function. It seems like there are ~40
    plates for which the bright/faint boundary is not near 17.8 mag for
    the K stars, but near 16 or 17 mag instead. A practical way to deal
    with this is to set the bright/faint boundary for each plate pair at
    the brightest spectroscopic K star on the faint plate of the pair.
    This seems to give an acceptable model for the K star selection
    function for the 'program' stars (those stars with the K-star target
    bit set); it does not work so well for the sample of all spectroscopic
    objects with 0.55 < g-r < 0.75, as there are always some stars much
    brighter than the interface. It's unclear whether this is a documented
    feature of the target selection (I cannot find any mention of it in
    the SEGUE paper).
