Python Astronomical Stacking Tool Array: PASTA
==============================================
## Introduction
PASTA is a software package written in the Python programming language for median
stacking of astronomical sources.  It includes a number of features for 
filtering sources, outputting stack statistics, generating Karma annotations, 
formatting sourcelists, and reading information from stacked Flexible Image 
Transport System (FITS) images.  PASTA was originally written to examine 
polarization stack properties, and it includes a Monte Carlo modeller for 
obtaining true polarized intensity from the observed polarization of a stack.  
PASTA is also useful as a generic stacking tool, even if polarization properties
are not being examined.

The basic operation of PASTA is to read in a sourcelist containing positions of 
sources to be stacked, as well as one or more FITS images.  PASTA then generates
an output list of stacked sources and their properties, and a pair of FITS 
files, one containing the median pixels of the stack, and the other containing 
the mean pixels.

Stacking allows the reduction of noise levels in the examination of sets of 
images.  It produces a pseudo-source consisting of the median of all the stacked
sources, with reduced background noise level.  For more information on the 
performance of stacking, see Stil et al. 2010.

## Software Requirements
PASTA uses a number of libraries to be installed prior to running.  Most of these
packages are available in the software repositories of the main Linux 
distributions.

* [Python 3.4+](http://www.python.org)
* [Numpy 1.7.0+](http://numpy.scipy.org)
* [Scipy 0.7.0+](http://www.scipy.org)
* [astropy 3.0.+](http://astropy.org)

## Using PASTA
For more details on how to build/use pasta, see the manual.

## Acknowledgements
PASTA was written by BW Keller, with scientific guidance from Jeroen Stil.  This
work was supported by an Discovery Grant from the Natural Sciences and
Engineering Research Council to Jeroen Stil.
