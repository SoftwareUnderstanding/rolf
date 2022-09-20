# Exoplanetary and Planetary Radio Emission Simulator (ExPRES) V1.1.0

<a href="http://ascl.net/1902.009"><img src="https://img.shields.io/badge/ascl-1902.009-blue.svg?colorB=262255" alt="ascl:1902.009" /></a>
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4292002.svg)](https://doi.org/10.5281/zenodo.4292002)
[![Documentation Status](https://readthedocs.org/projects/expres/badge/?version=latest)](https://expres.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


## Introduction
The ExPRES models CMI (Cyclotron Maser Instability) driven radio emissions. It provides radio dynamic spectra observed from a defined location. Since the CMI emission process is very anisotropic, the relative geometry of the radio source and the observer drives the observability of the source. More info on the ExPRES code: [ExPRES on the MASER web site](http://maser.lesia.obspm.fr/tools-services-6/expres/). The code can be launched from [the MASER run on demand interface](https://voparis-uws-maser.obspm.fr). 

Reference: [Louis et al., 
ExPRES: a Tool to Simulate Exoplanetary and Planetary Radio Emissions, A&A 627, A30 (2019)](https://doi.org/10.1051/0004-6361/201935161) 

## Repository Description 

### Directories
* [src](src) contains the ExPRES code IDL routines.
* [mfl](mfl) stores the magnetic field lines used by ExPRES. When installing the code, precomputed data 
files must be retrieved from [http://maser.obspm.fr/support/expres/mfl](http://maser.obspm.fr/support/expres/mfl). 
See [this file](mfl/README.md) for more details.
* [ephem](ephem) stores ephemerides files used by ExPRES. IDL saveset files (.sav) are available for precomputed
ephemerides. Other files (plain text format, .txt) will be stored here, and correspond to the output of the 
MIRIADE IMCCE webservice calls.
* [cdawlib](cdawlib) is a placeholder for the NASA/GSFC CDAWLib library, required for the CDF files. 

### Configuration
The `config.ini.template` file (in [src](src)) must be renamed `config.ini` and edited with the adequate paths:
- `cdf_dist_path` must point to the local CDF distribution directory.
- `ephem_path` is the path to the directory where the precomputed ephemerides files are
located, and where temporary ephemerides files will be written.
- `mfl_path` is the path to the directory where the precomputed magnetic field line data.
- `save_path` is the path where the data will be saved.
- `ffmpeg_path` points to the `ffmpeg` executable, e.g., `/opt/local/bin/ffmpeg`
- `ps2pdf_path` points to the `ps2pdf` executable, e.g., `/opt/local/bin/ps2pdf`

Examples are provided in the header of [config.ini.template](src/config.ini.template).

### Running the code
The code has been tested under IDL 8.5. You must have a functional installation of IDL 8.5 (or better) on your system.

The IDL interpreter must be configured to look for routines into the [src](src) and [cdawlib](cdawlib) directories.

The operation are initiated with the following batch script:
```
IDL> @serpe_compile
``` 
This compiles all the necessary routines in advance. Then the simulation can be launched:
```
IDL> main,'file.json'
```
where `file.json` is the input parameter file. This file must comply with the [ExPRES-v1.1 JSON-shema](https://voparis-ns.obspm.fr/maser/expres/v1.1/schema#)

