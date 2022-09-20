Reversible Jump Objects
=======================

Classes for generic birth/death MCMC in
[DNest3](https://github.com/eggplantbren/DNest3).

(c) 2013-2015 Brendon J. Brewer

Licence: GNU GPL v3 for software, CC BY-NC-SA for documents.

This work is supported by a Marsden Fast-Start grant
from the Royal Society of New Zealand.

News
====

A significant bug was fixed at the beginning of September 2015. If you use
RJObject, please make sure your copy of this repository is up to date.


Installation
============

Building RJObject requires that you have the following packages installed:

* [GNU Scientific Library (GSL)](http://www.gnu.org/software/gsl/)
* [Boost](http://www.boost.org/)
* [CMake](http://www.cmake.org/)
* [DNest3](https://github.com/eggplantbren/DNest3)

The first three can be most conveniently installed using your favourite
package management system (e.g. [Homebrew](http://mxcl.github.com/homebrew/)
or [Macports](https://www.macports.org/) on a Mac, APT on
[Ubuntu](http://www.ubuntu.com/) or [Debian](http://www.debian.org/)). DNest3
should be compiled and installed following its documentation.

You can then build RJObject as follows:

```
git clone https://github.com/eggplantbren/RJObject.git
mkdir RJObject/build
cd RJObject/build
cmake ..
make
make install
```

This will install the RJObject library and header files into `/usr/local`,
while example programs are available in `RJObject/Examples`. If required,
change the installation location by giving a `-DCMAKE_INSTALL_PREFIX` option
to CMake:

```
cmake -DCMAKE_INSTALL_PREFIX=/tmp ..
```

This procedure assumes that DNest3 has been installed into a standard system
location. If you have installed it elsewhere (say, `/tmp/dnest`), specify the
location as follows:

```
cmake -DNEST3_ROOT_DIR=/tmp/dnest ..
```

It is also possible to build against a copy of DNest3 which has not been
installed (but has, of course, been compiled) by directly specifying the
location of the library and headers. Assuming it has been downloaded and built
(as a static library, which is the default) in `/home/username/DNest3`:

```
cmake -DDNEST3_INCLUDES=/home/username/DNest3/include -DDNEST3_LIBRARY=/home/username/DNest3/build/libdnest3.a ..
```

Acknowledgements
================

Thanks to Stephen Portillo (Harvard) for finding a bug.

