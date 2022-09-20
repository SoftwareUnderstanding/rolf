# nrm_analysis README #

Reduces aperture masking images to fringe observables, calibrates, does basic model fitting. Package development led by Alexandra Greenbaum following legacy code by Greenbaum, Anand Sivaramakrishnan, and Laurent Pueyo. Contributions from Sivaramakrishnan, Deepashri Thatte, and Johannes Sahlmann.

To get the source files:

	git clone https://github.com/agreenbaum/nrm_analysis.git

Necessary Python packages:

* numpy
* scipy
* astropy
* matplotlib
* a copy of [Paul Boley's oifits.py](http://astro.ins.urfu.ru/pages/~pboley/oifits/) in your python path
* emcee [available here](http://dan.iel.fm/emcee/current/) or by pip install
* DFM's corner [source here](https://github.com/dfm/corner.py) or by pip install
* uncertainties package [information and instruction here](https://pythonhosted.org/uncertainties/)
* linearfit [available here](https://pypi.org/project/linearfit/) and by pip install

Optional Python packages:

* webbpsf [available here](https://github.com/mperrin/webbpsf) or by pip install
* poppy [available here](https://github.com/mperrin/poppy) or by pip install
* pysynphot

*we recommend downloading the anaconda distribution for python*

**For usage examples check the notebooks folder, which contains some basic tutorials.**

### Core routines: ###

* **FringeFitter** - fit fringes in the image plane. Support for masks on GPI, VISIR, and JWST-NIRISS
* **Calibrate** - calibrate raw frame phases, closure phase, and squared visibilities, save to oifits option
* **Analyze** - support for basic binary model fitting of calibrated oifits files

### Using this package ###

**Step 1**: *fit fringes in the image plane.* 

initializes NRM_model & generates best model to match the data (fine-tuning
	  centering, scaling, and rotation). Fits the data, saves to text files.
	  This driver creates a new directory for each target and saves the 
	  solution files described in the driver for each datacube slice.

**Step 2**: *calibrate the data.* 

User defines target and calibration sources in a list [target, cal, cal, ...], pointing to the directories containing each source's measured fringe observables computed from driver 1. Calculates mean and standard error, calibrates target observables with calibrator observables and does basic error propogation. Saves calibrated quantities (recommended save to oifits) to specified directory.

**Step 3**: *compare the data to a model (3 options).*

Currently there is basic framework to fit the model of a binary or multiple point source. The BinaryAnalyze module accepts a calibrated oifits file and has several useful tools for exploring the data:

1. _Coarse search_ - the chi2map method calculates chi^2 over a coarse range of parameters, saves a grid in position at the contrast that minimizes chi^2 at each postion, and returns a first guess for the mcmc fit
2. _"correlation" plot_ - plots measured closure phases against model closure phases, tunable parameter -- a handy tool to visualize the data
3. _mcmc_ - using the **emcee** package can find the best fit binary/multiple source model to the measured fringe observables

## Basic tutorial ##
The main modules in this package are *InstrumentData* and *nrm_core*. **InstrumentData** sets up your dataset based on the instrument you're working with. **nrm_core** contains *FringeFitter*, *Calibrate*, and *BinaryAnalyze*, described above. Here are basic examples of how to run this package with NIRISS or GPI data.


###NIRISS Example ###
Start by importing main package modules. There is example data provided in this package that you can load to try this demo.

	from nrm_analysis import InstrumentData, nrm_core
	
	# define data files
	import os
	targfiles = [f for f in os.listdir("f430_data") if "tcube" in f] # simulated data
	calfiles = [f for f in os.listdir("f430_data") if "ccube" in f] # simulated data
	
	# set up instrument-specfic part
	nirissdata = InstrumentData.NIRISS(filt="F430M", objname="targ")

This last call is an instance of 'NIRISS', which will set up the data according to NIRISS standards given a filter name and the name of the observed object. We are working with the test data provided, which includes images for 1 target and 1 calibrator:

	ff =  nrm_core.FringeFitter(nirissdata, oversample = 5, savedir="targ", datadir="f430_data", npix=75)
	ff.fit_fringes(targfiles)
		
	ff2 =  nrm_core.FringeFitter(nirissdata, oversample = 5, savedir="cal1", datadir="f430_data", npix=75)
	ff2.fit_fringes(calfiles)

	
The fringe fitter (with the options you want) measures fringe observables - visibilities and closure phases. It fits each exposure by calling the fit_fringes method and saves the output to directories "targ" and "cal1." The default is to save to the working directory. I usually set savedir to a string with the object's name. FringeFitter, creates a subdirectory named after each fits file name passed. So if you provide a single image per fits file (as in this example) then there will be one FF reduction per new subdirectory created. If you fit a cube of images then the reduction of each slice will be saved in a directory corresponding to the file name. 

	targdir = "targ/"
	caldir = "cal1/"	
	calib = nrm_core.Calibrate([targdir, caldir], nirissdata, savedir = "my_calibrated", sub_dir_tag="cube")

Instance of Calibrate, gives 2 directories containing target and calibration sources. The first directory in the list is always assumed to be the science target. Any number of calibrators may be provided. Argument savedir default is "calibrated."  In example I provide, each exposure is a single fits file, so . Argument sub_dir_tag is the common part of each file name. If you choose to analyze a cube of NIRISS images, you should not set a "sub_dir_tag" argument, since each slice is a new exposure. If interactive is on (default) then it will warn you about this. (See below in GPI example for an alternative example, where the cube contains a wavelength axis).

	calib.save_to_oifits("niriss_test.oifits")
Saves results to oifits. phaseceil keyword arg optional to set a custom dataflag. Default flag is set when phases exceed  1.0e2. 

*Keyword argument "interactive" in FringeFitter and Calibrate is by default set to True, which checks before overwriting directory contents and unusual settings. Set it to False if you know what you're doing and you want to speed up the process.*

Now that you have calibrated data and you suspect there is a binary companion, you can try out a few different routines an instance of BinaryAnalyze using the oifits file created in the last step. Let's start with a coarse search within a range of reasonable parameters. 

	# Initialize binary model with the oifits file you want to analyze
	dataset = nrm_core.BinaryAnalyze("my_calibrated/niriss_test.oifits", savedir="my_calibrated/") 
	# as before you can specify a custom savedir argument for where you want to store the results

	# set some bounds for the search
	bounds = [(0.001, 0.99), (40, 200), (0, 360)] # contrast ratio, separation (mas), and pa (deg)
	
	dataset.coarse_binary_search(bounds, nstep=25) # set the 'resolution' of the search with nstep. default is nstep=20
	
	# Now I have a new routine that plots a map the minimum chi^2 contrast at each position
	dataset.chi2map()	
	# If this is taking too long, make nstep smaller (default is nstep=50]). If you want more detail increase it. Default threads is 4. To remove parallel processing set threads=0.

	# Does it look like you have a single, strong minimum?

(1) will plot the likelihood over a course grid for pairs of parameters and prints the location of the highest likelihood. (2) will plot the minimum chi^2 at each position in a coarse grid, and print/return the minimum chi^2 parameters. Either way, let's call these values c_val, s_val, and p_val (for contrast, separation, and pa). We can plug these 'close' guesses into the mcmc method:

	params = {'con':c_val, 'sep':s_val, 'pa':p_val} # give params as a dictionary
	priors = [(0.0001, 0.999), (10, 500), (0, 360)] # set some bounds on the search

	# method run_emcee uses dfm's emcee package to search for the location and relative brightness of a binary source. 
	dataset.run_emcee(params, nwalkers=200, niter=1000, priors=priors, scale=np.sqrt(7/3.0)) # default nwalkers is 250, niter is 1000
	# The last kwarg is a scaling factor to multiply by the error accounting for the ratio between total and independent closure phase (in this case 35 total, 15 independent)
	# Triangle plot:
	dataset.corner_plot() # it will also save "triangle_plot.pdf" to your savedir
	# if you want to see how the chains have behaved you can plot them easily:
	dataset.plot_chain_convergence()
	# plots will be saved to the 'savedir' you set (default 'calibrated/')

### GPI Example ###
e.g., starting with a list of GPI files

	from nrm_analysis import InstrumentData, nrm_core
	
	gpifiles = [S20130501S00{0:02d}_spdc.fits.format(q) for q in np.arange(10)] # list files in here, e.g., spectral datacubes from May 1 2013.

	gpidata = InstrumentData.GPI(gpifiles) # will pull some relevant observation parameters from the headers, such as parallactic angle. 

This creates an instance of GPI, which will read header keywords from a reference file and sets up the data according to GPI standards


	ff =  nrm_core.FringeFitter(gpidata, oversample = 5, savedir="Target", npix=121) for exposure in gpifiles:
		ff.fit_fringes(exposure)

This initializes the fringe fitter with the options you want for measuring fringe observables. Then it fits each fits file's data by calling the fit_fringes method. Saves the output to directory "Target/" in working directory. I usually name this by the object name. 

	targdir = "Target/"
	caldir = "Calibrator1/"
	cal2dir = "Calibrator2/"
	calib = nrm_core.Calibrate([targdir, caldir, cal2dir], gpidata, savedir = "my_calibrated", sub_dir_tag = "130501")


Instance of Calibrate, gives 3 directories containing target and any calibration sources. The first directory in the list is always assumed to be the science target. Any number of calibrators may be provided. Argument savedir default is "calibrated." Argument sub_dir_tag must be provided if there is an additional axis (multiple wavelengths, or pollarizations), to save results from each slice into sub directories separated by exposure.
 
	calib.save_to_oifits("targ_vis.oifits", phaseceil = 50.0)
Saves results to oifits. phaseceil keyword arg can be used to flag phases (> 50.0 degrees in this case). Default is 1.0e2.
