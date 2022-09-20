# HISS
HI Stacking Software

This package has a couple of Python dependencies:
* SciPy (tested with version 0.17.0)
* Numpy (tested with version 1.9.1)
* Astropy (tested with version 1.1.2)
* PyYAML [this is a dependency of Astropy] (tested with version 3.11)

If any of the above packages are missing, you can use:

	`sudo pip install <package name>`

Note: this package has been developed and tested on a computer running MacOS Yosemite and above using Python 2.7.9 as well as Python 3.6.

FIRST TIME USERS:
	It is recommended that you use the graphical interface to populate a config file. The graphical interface is called by the "hiss" executable.

To use the stacker from the command-line, enter the following command:

	`python pipeline.py [-h] [-f <filepath+filename>] [-d] [-p] [-s] [-c] [-l]`

There are a number of different options that can be used to run this package:

optional arguments:

	-h, --help            show this help message and exit.
	-d, --display         Option to display progress window during the stacking process.
	-p, --saveprogress    Option to save progress window during the stacking process.
	-s, --suppress        Use this flag to suppress all output windows. Note that [suppress] and [progress] cannot be used simultaneously.	
	-c, --clean           Use [clean] for testing purposes and for stacking noiseless spectra as this option will bypass any noise-related functions and actions.	
	-l, --latex           This option enables to the use of latex formatting in the plots.                    

An example configuration file is included with the modules that can be edited and used instead 
of manually entering the input information.

The data created by the Stacker include:
  1. OutputData.FITS:       This file contains the stacked spectrum, reference spectrum, fitted spectrum parameters and stacked noise.
  2. IntegratedFlux.csv:    A table containing the integrated flux calculated from the  fitted functions and the stacked spectrum.
  3. StackedSpectrum.pdf:   A plot of the stacked spectrum and reference spectrum along with any fitted functions.
  4. NoiseAnalysis.pdf:     A plot of the stacked noise response.
  5. Stacked_Catalogue.csv: A table containing the catalogue information of all the spectra included in the stacked spectrum.
  6. FluxDistribution.pdf:  A plot of the distribution of the integrated fluxes - this plot is only created when calculating the uncertainties.
  7. Log file               This is a detailed log file of the stacking procedure.

