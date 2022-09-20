AUM
===

AUM: A Unified Modelling scheme for galaxy abundance, galaxy clustering and
galaxy-galaxy lensing.


This piece of software has general purpose routines that allow prediction of
the galaxy abundances, their clustering and the galaxy-galaxy lensing signal, given the halo occupation distribution of galaxies and the underlying cosmological model. 

In combination with the measurements of the clustering, abundance and lensing
of galaxies, these routines can be used to perform parameter inferences.

In order to install the code, please install 
a) C++ compiler, 
b) swig, (http://www.swig.org)
c) GNU Scientific Library (http://www.gnu.org/software/gsl/)

Make sure to put the path to swig, g++, in your PATH environment variable, the
path to the gsl include files in your includes and the path to the gsl library 
in your LD_LIBRARY_PATH variable and LDFLAGS.

For example,
```bash
export LDFLAGS=-L`gsl-config --prefix`/lib
export CPPFLAGS=-I`gsl-config --prefix`/include
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`gsl-config --prefix`/lib
```

Then for a quick compilation of the python modules:
```bash
python setup.py install --prefix=`pwd`/install
python setup.py install --prefix=`pwd`/install
```
Yes you have to run the same command twice, so that the swig generated python
modules are also copied correctly.

If all goes well, you should have a working python library in the subdirectory
install/lib/python3.x/...

Note the path that is output by the following command:
```bash
echo `pwd`/`find install -iname site-packages`
```

To test the installation, run:

```python
import sys
sys.path.append('PATH_OUTPUT_BY_PREVIOUS_COMMAND')
import cosmology as cc

# This is the default constructor with some basic cosmological parameters
a=cc.cosmology()
# Prints out the comoving distance in the fiducial cosmology
print a.Dcofz(2.0)

# Prints the abundance of 1e9 Msun halos at z=0.0
print a.nofm(1e9,0.0)

# Print all functions
help(a)
```

