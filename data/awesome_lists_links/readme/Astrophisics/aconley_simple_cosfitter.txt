simple_cosfitter
================

This is a supernova-centric grid based fitter for measuring
the cosmological parameters, specifically for fitting the
dark energy equation of state.


###Installation
Installation is via the standard UNIX `configure` and
`make`. pofd_affine depends on the following packages:
* [cfitsio](http://heasarc.gsfc.nasa.gov/fitsio/)
* The [GNU scientific library](http://www.gnu.org/software/gsl/)
* It can benefit significantly from LAPACK/BLAS libraries,
   either [ATLAS](http://math-atlas.sourceforge.net/) or
   the [Intel MKL](http://software.intel.com/en-us/articles/intel-mkl/);
   on OSX there is also support for the built in accelerate libraries.
It may be necessary to tell configure where to look for these
libraries -- see `configure --help`.

Note that if you got this from github rather than downloading
a .tgz, you may need to do something like `autoreconf -fvi` before
building to set up the build environment.

### Documentation

Fuller documentation is provided using Doxygen.  This
can be generated using

	make docs

The resulting documentation describes the format of the
input file.

### Branches

The most interesting branch -- which will never be merged --
is a branch to handle the galaxy dependent offset in the absolute
magnitude of Type Ia SNe as described in
[Sullivan et al. (2010)](http://adsabs.harvard.edu/abs/2010MNRAS.406..782S).
This is the twoscripm branch, and was the one actually used in the SNLS
3rd year analyses.

### References
* This code is based on one I wrote for my PhD thesis, as
  presented in [Conley et al. 2006](http://adsabs.harvard.edu/abs/2006ApJ...644....1C)
* It was used in the SNLS 3rd year analysis in both
  [Conley et al. 2011](http://adsabs.harvard.edu/abs/2011ApJS..192....1C)
  and [Sullivan et al. 2011](http://adsabs.harvard.edu/abs/2011ApJ...737..102S)
* It was also used in the ESSENCE results, as described in
  [Wood-Vasey et al. (2007)](http://adsabs.harvard.edu/abs/2007ApJ...666..694W)
