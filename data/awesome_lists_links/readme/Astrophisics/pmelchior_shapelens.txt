shapelens
=========

The C++ library provides ways to load galaxies and star images from FITS files and catalogs and to analyze their morphology. The main purpose of this library is to make several weak-lensing shape estimators publicly available. All of them are based on the moments of the brightness distribution, several of them have been used in practice, some others are of more theoretical interest:

* `DEIMOS`: Analytic deconvolution in moment space. Introduced by [Melchior et al. (2011)](http://adsabs.harvard.edu/abs/2011MNRAS.412.1552M).
* `DEIMOSElliptical`: A practical implemention of `DEIMOS` with an automatically matched elliptical weight function. The matching procedure iteratively determines centroid, size, and ellipticity of the weight function such as to maximize the measurement S/N. 
* `DEIMOSCircular`: Identical to `DEIMOSElliptical` but with a circular weight function.
* `KSB`: Based on [Kaiser et al. (1995)](http://adsabs.harvard.edu/abs/1995ApJ...449..460K) with modifications by [Viola et a. (2011)](http://adsabs.harvard.edu/abs/2011MNRAS.410.2156V) and a matching procedure identical to the one in `DEIMOSCircular`.
* `HOLICS`: A generalization of `KSB` to flexions, the next-higher order lensing shape measurements. Introduced by [Okura et al. (2007)](http://adsabs.harvard.edu/abs/2007ApJ...660..995O).

If you use these algorithms for your work, please do not forget to cite the relevant papers.

Installation
------------

The library should work under Linux and Mac OS X, with any reasonably recent C++ compiler.

### Prerequisites
* [Boost](http://www.boost.org/), headers only
* [tmv](http://code.google.com/p/tmv-cpp/) for fast matrix/vector operations (v0.71+; make sure you compile with the `scons` option `INST_INT=true`, see #1)
* [cfitsio](http://heasarc.gsfc.nasa.gov/fitsio/) to work with FITS files, the ubiquitous file format in astronomy

For the World Coordinate System (WCS, to relate pixel coordinates to coordinates on the sky), please use [libwcs](http://www.atnf.csiro.au/people/mcalabre/WCS/index.html).

### Compilation option

Compilations is done with a Makefile, so calling `make` in the main directory should do the trick. 

The Makefile recognizes several environment variables:
* `PREFIX`: where the library and headers are to be installed (default: `/usr/`)
* `TMV_PREFIX`: where the tmv library can be found (default: `$PREFIX`)
* `SPECIAL_FLAGS`: additional compiler flags to optimize output and provide include directories, e.g.
  `-03 -m64 -march=native -fPIC`. Make sure you add the flag `-DNDEBUG` for a production system.
* `SPECIAL_LIBS`: additional linker flags and directories, e.g.
  `-L$HOME/lib`

If the WCS library should be used, the flag `-DHAS_WCSLIB` needs to be set in `SPECIAL_FLAGS`.

`make install` copies the static and shared library to `$PREFIX/lib` and the headers to `PREFIX/include/shapelens`. `make progs` compiles any `.cc` file in the `progs` directory, `make installprogs` copies the executables to `$PREFIX/bin`.

`make clean` and `make cleanprogs` delete all compiled code. 

Documentation
-------------

`make docs` runs [Doxygen](http://www.stack.nl/~dimitri/doxygen/index.html) on the headers, the files go to `doc/html`. An online version of the latest development branch is available [here](http://www.physics.ohio-state.edu/~melchior.12/docs/shapelens/classes.html).

License (MIT)
-------------

Copyright (c) 2012, Peter Melchior

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


