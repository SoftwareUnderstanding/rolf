#  MIRIAD

MIRIAD (Multi-channel Image Reconstruction, Image Analysis, and
Display) was originally developed for BIMA, and has been adopted and
expanded for a number of radio telescopes arrays (CARMA, SMA, WSRT,
ATNF and perhaps more).  Sadly each of these have cloned and diverged
from the original version of MIRIAD. So be it. What you see here is
the original BIMA/CARMA version, as originally developed by Bob Sault
in the late 80s and was actively used for CARMA until 2015, and is still
in some use.

# Reference

The original Sault et al. (1995) paper is https://ui.adsabs.harvard.edu/abs/1995ASPC...77..433S
where you can also find the bibtex entry:

      @INPROCEEDINGS{1995ASPC...77..433S,
             author = {{Sault}, R.~J. and {Teuben}, P.~J. and {Wright}, M.~C.~H.},
              title = "{A Retrospective View of MIRIAD}",
           keywords = {Astrophysics},
          booktitle = {Astronomical Data Analysis Software and Systems IV},
               year = 1995,
             editor = {{Shaw}, R.~A. and {Payne}, H.~E. and {Hayes}, J.~J.~E.},
             series = {Astronomical Society of the Pacific Conference Series},
             volume = {77},
              month = jan,
              pages = {433},
      archivePrefix = {arXiv},
             eprint = {astro-ph/0612759},
       primaryClass = {astro-ph},
             adsurl = {https://ui.adsabs.harvard.edu/abs/1995ASPC...77..433S},
            adsnote = {Provided by the SAO/NASA Astrophysics Data System}
      }



# Installation

We have a csh script **[install_miriad](docs/install_miriad)** and a much simpler bare bones
**[install_miriad.sh](docs/install_miriad.sh)**. They may contain helpful comments to get you
past some hurdles, but here are briefly the steps on a linux machine,
extracted from those scripts:

      git clone https://github.com/astroumd/miriad
      cd miriad
      install/install.miriad  gfortran=1  generic=1  gif=1  telescope=carma

This installation will take about 5 minutes, and usually takes up 300-400MB.

# Requirements

The following tools should be present:  a Fortran and C compiler, make,
csh, git, development libraries for X11, optionally automake and pgplot library.
It is recommended to use the native pgplot, not the pgplot that miriad contains,
because the former now supports PNG's.

## Ubuntu

Essentials:

      sudo apt install git tcsh build-essential gfortran xorg-dev libreadline6-dev -y

Optionals:

      sudo apt install pgplot5 automake libtool flex -y

## Centos

Native pgplot needs a special install. Not tested.

# History

* V1 Original BIMA - Bob Sault (1987-1990)
* V2 BIMA - RCS based (1990-2000)
* V3 BIMA and CARMA - CVS based (2001-2003)
* V4 BIMA and CARMA - 64 bit  (2003-2016)
* V5 - never released -
* V6 CARMA - github based, pgplot decoupled
