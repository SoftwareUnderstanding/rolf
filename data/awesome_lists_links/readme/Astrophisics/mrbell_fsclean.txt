fsclean - Faraday Synthesis CLEAN imager
==========================================================

fsclean is a software package for producing 3D Faraday spectra using the Faraday
synthesis method, transforming directly from multi-frequency visibility data
to the Faraday depth-sky plane space. Deconvolution is accomplished using the
CLEAN algorithm.

Features include: 

  - Reads in MeasurementSet visibility data.
  - Clark and Högbom style CLEAN algorithms included.
  - Produces HDF5 formatted images. Simple matplotlib based visualization tools 
    are planned.
  - Handles images and data of arbitrary size, using scratch HDF5 files as 
    buffers for data that is not being immediately processed. Only limited by
    available disk space.

For more information, see the Faraday synthesis paper [(Bell & Enßlin, 2012)](http://adsabs.harvard.edu/cgi-bin/bib_query?arXiv:1112.4175).

Installation and Usage
----------------------

Prerequisites for fsclean
  - python
  - numpy
  - matplotlib (for the plotting_tools.py module)
  - [pyrat](https://github.com/mrbell/pyrat)
  - cython
  - gsl

Before one can use fsclean, some code needs to be compiled using the command

   > python setup.py build_ext --inplace

You may have to edit the include_gsl_dir and lib_gsl_dir variables within the
setup.py file to point to the appropriate GSL header and library file 
directories for your system.

Usage instructions can be obtained by typing

   > fsclean.py -h

and a description of the parameter file options is given by typing
  
   > fsclean.py -p

About fsclean
-------------

fsclean is licensed under the [GPLv3](http://www.gnu.org/licenses/gpl.html).

fsclean has been developed at the Max Planck Institute for Astrophysics and 
within the framework of the DFG Forschergruppe 1254, "Magnetisation of 
Interstellar and Intergalactic Media: The Prospects of Low-Frequency Radio
Observations."
