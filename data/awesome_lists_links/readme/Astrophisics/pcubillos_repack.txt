# repack
Repack and Compress Line-transition Data for Radiative-tranfer Calculations

[![Build Status](https://travis-ci.com/pcubillos/repack.svg?branch=master)](https://travis-ci.com/pcubillos/repack)
[![PyPI](https://img.shields.io/pypi/v/lbl-repack.svg)](https://pypi.org/project/lbl-repack)
[![GitHub](https://img.shields.io/github/license/pcubillos/repack.svg?color=blue)](https://github.com/pcubillos/repack/blob/master/LICENSE)

This code identifies the strong lines that dominate the spectrum from
the large-majority of weaker lines.  The code returns a binary
line-by-line (LBL) file with the strong lines info (wavenumber, Elow,
gf, and isotope ID), and an ascii file with the combined contribution
of the weaker lines compressed into a continuum extinction coefficient
(in cm-1 amagat-1) as function of wavenumber and temperature.

Currently available databases:
* ExoMol (http://www.exomol.com/)
* HITRAN (https://www.cfa.harvard.edu/hitran/)
* Kurucz's TiO (http://kurucz.harvard.edu/molecules/tio)

### Team Members
* [Patricio Cubillos](https://github.com/pcubillos/) (IWF) <patricio.cubillos@oeaw.ac.at>

### Install
``repack`` has been [tested](https://travis-ci.com/pcubillos/repack) to work on Python 3.6 and 3.7; and runs (at least) in both Linux and OSX.  You can install ``repack`` from the terminal with pip:

```shell
# Note that on PyPI ``repack``is indexed as ``lbl-repack``:
pip install lbl-repack
```

Alternative (for developers), you can directly dowload the source code
and install to your local machine with the following terminal commands:

```shell
git clone https://github.com/pcubillos/repack/
cd repack
python setup.py install
```

### Getting Started

The following example compresses the Exomol HCN line-transition data.  First, download the ExoMol HCN dataset (there is no need to unzip the files):

```shell
# Download ExoMol HCN data:
wget http://exomol.com/db/HCN/1H-12C-14N/Harris/1H-12C-14N__Harris.states.bz2
wget http://exomol.com/db/HCN/1H-12C-14N/Harris/1H-12C-14N__Harris.trans.bz2
wget http://exomol.com/db/HCN/1H-12C-14N/Harris/1H-12C-14N__Harris.pf
wget http://exomol.com/db/HCN/1H-13C-14N/Larner/1H-13C-14N__Larner.states.bz2
wget http://exomol.com/db/HCN/1H-13C-14N/Larner/1H-13C-14N__Larner.trans.bz2
wget http://exomol.com/db/HCN/1H-13C-14N/Larner/1H-13C-14N__Larner.pf
```

Then create a repack configuration file ('*repack_HCN.cfg*') like this below:

```shell
[REPACK]

# Line-transition files:
lblfiles = 1H-12C-14N__Harris.trans.bz2
           1H-13C-14N__Larner.trans.bz2

# Database type [exomol, hitran, or kurucz]:
dbtype = exomol

# Output file name (without file extension):
outfile = HCN_exomol_harris-larner_0.3-33um_100-3000K_sthresh_0.01

# Wavenumber boundaries and sampling rate (in cm-1):
wnmin =   303.0
wnmax = 33334.0
dwn   =     1.0

# Temperature sampling:
tmin  =  100.0
tmax  = 3000.0
dtemp =  100.0

# Line-intensity threshold for strong/weak lines:
sthresh = 0.01

# Maximum chunk size of lines to handle at a time:
chunksize = 5000000
ncpu = 5
```

And run ``repack``, which will produce the following screen output:
```shell
# Call the repack command-line executable for the HCN demo config file:
repack repack_HCN.cfg

::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  repack: line-transition data compression.
  Version 1.4.1.
  Copyright (c) 2017-2020 Patricio Cubillos.
  repack is open-source software under the MIT license.
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


Starting: Fri Apr  3 14:45:25 2020
Unzipping: '1H-12C-14N__Harris.trans.bz2'.
Unzipping: '1H-13C-14N__Larner.trans.bz2'.
Reading: '1H-12C-14N__Harris.trans.bz2'.
Reading: '1H-13C-14N__Larner.trans.bz2'.
  Flagging lines at  100 K (chunk 1/14):
  Compression rate:       96.82%,    148,115/ 4,662,663 lines.
  Flagging lines at 3000 K:
  Compression rate:       86.89%,    611,256/ 4,662,663 lines.
  Total compression rate: 84.60%,    717,921/ 4,662,663 lines.

...

  Flagging lines at  100 K (chunk 14/14):
  Compression rate:       95.47%,    209,217/ 4,619,175 lines.
  Flagging lines at 3000 K:
  Compression rate:       75.13%,  1,148,804/ 4,619,175 lines.
  Total compression rate: 73.22%,  1,237,122/ 4,619,175 lines.

With a threshold strength factor of 0.01,
kept a total of 7,553,671 line transitions out of 65,586,274 lines.

Successfully rewriten exomol line-transition info into:
  'HCN_exomol_harris-larner_0.3-33um_100-3000K_sthresh_0.01_lbl.dat' and
  'HCN_exomol_harris-larner_0.3-33um_100-3000K_sthresh_0.01_continuum.dat'.
End: Fri Apr  3 14:51:06 2020
```

The output binary file '*HCN_exomol_harris-larner_0.3-33um_100-3000K_sthresh_0.01_lbl.dat*'
contains the line-by-line opacity information for HCN, which represent
most of the opacity contribution into the spectrum.  The information
is encoded as a sequence of three doubles and an integer containing
the wavenumber (in cm-1), lower-state energy (in cm-1 units),
gf value, and isotope index, respectively, for each transition.  This
info can be easily read with the following python script:

```python
import repack.utils as u
wn, elow, gf, iiso = u.read_lbl('HCN_exomol_harris-larner_0.3-33um_100-3000K_sthresh_0.01_lbl.dat')
```

The output ascii file '*HCN_exomol_harris-larner_0.3-33um_100-3000K_sthresh_0.01_continuum.dat*'
contains the remaining opacity contribution of the weak lines (in cm-1
amagat-1 units) as function of wavenumber and temperature.  This is a
minor contribution compared to that of the LBL output file.


### Re-sorting MARVELized files

Since some ExoMol .states files have been MARVELized (refined energy levels), the .trans files are no longer sorted by wavenumber.  This is a problem for ``repack`` since its binaary searches rely on a sorted wavenumber files.  To solve this, the user should sort the files before repacking:

```shell
# First sort the .trans files (use same config file as a repack file):
repack -sort repack_H2O.cfg

# Now run repack as usual:
repack repack_H2O.cfg
```


### Be Kind

Please, be kind and acknowledge the effort of the authors by citing the article asociated to this project:  

  [Cubillos (2017): An Algorithm to Compress Line-transition Data for Radiative-transfer Calculations](http://adsabs.harvard.edu/abs/2017ApJ...850...32C), ApJ 850, 32.  


### License

Copyright (c) 2017-2020 Patricio Cubillos.
``repack`` is open-source software under the MIT license (see [LICENSE](https://github.com/pcubillos/repack/blob/master/LICENSE)).

