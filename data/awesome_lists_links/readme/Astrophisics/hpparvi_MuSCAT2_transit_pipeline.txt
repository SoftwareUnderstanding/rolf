# MuSCAT2 photometry and transit analysis pipelines

Python-based photometry and transit analysis pipelines for MuSCAT2 developed in collaboration with the [Instituto de Astrofísica de Canarias (IAC)](http://www.iac.es), University of Tokyo (UT), [National Astronomical Observatory of Japan (NAOJ)](http://www.nao.ac.jp), [The Graduate University for Advanced Studies (SOKENDAI)](http://guas-astronomy.jp), and [Astrobiology Center (ABC)](http://abc-nins.jp).

The pipeline consists of a set of executable scripts and two Python packages: `muscat2ph` for photometry, and `muscat2ta` for transit analysis. The MuSCAT2 photometry can be carried out using the scripts only. The transit analysis can also in most cases be done using the main transit analysis script `m2fit`, but the `muscat2ta` package also offers high-level classes that can be used to carry out more customised transit analysis as a Python script (or Jupyter notebook).

## Overview

### Photometry

MuSCAT2 photometry pipeline consists of three steps, each carried out by an executable script

  1. `m2organize` organises the observed raw frames into a logical directory hierarchy,
  2. `m2astrometry` calculates the astrometry for the science frames using `astrometry.net`,
  3. `m2photometry` calculates the aperture photometry.

The first two steps are meant to be run on per-night basis. First, `m2organize` is used to organise the frames from one night of observations in a directory `yymmdd`, into a directory hierarchy where the calibration and science frames are separated, different objects are separated, and different CCDs are separated as

    yymmdd
    |- calib
     |- flat
      |- [g, r, y, z]
     |- dark
      |- [g, r, y, z]
    |- obj
     |- obj1name
      |- [g, r, y, z]
     |- obj2name
      |- [g, r, y, z]
     |- objnname
      |- [g, r, y, z]

*Note that `m2organize` does not delete or overwrite the original raw data, but creates a new copy for each night*.

Next, `m2astrometry` is ran on the organised directory. The script is a parallel wrapper over `astrometry.net` that calculates the astrometric solution for each science frame `filename.fits`, and stores it in a separate `filename.wcs` sidecar file.

Finally, `m2photometry` is ran on the organised directory to calculate either the photometry for all the targets observed during a single night, or for a single target if more customised photometry is required.


### Transit Analysis
The transit analysis pipeline offers an executable script `m2fit` that can be used to carry out multicolour transit analysis, and also a set of high-level classes that make writing customised transit analysis scripts straightforward.

Transit analysis consists of a set of basic steps

1. Define the problem
1. Select the optimal comparison stars and the optimal target and comparison apertures
2. Fit a transit using a linear systematics model (LM)
3. Learn the GP hyperparameters from the data after removing the LM transit
1. Fit a transit using a GP systematics model (GPM)
2. Sample the GP systematics model posterior using MCMC
3. Plots the light curves, systematics, and parameter posteriors
4. Save the posterior samples and the final light curves

The `m2fit` scripts carries out the steps and offers some customisability through command line options. However, the pipeline also comes with a Jupyter notebook template that carries out the steps with significantly higher level of interaction and customisability than offered by the script.

## Requirements

 - Python 3.6
 - NumPy, SciPy, Pandas, xarray, scikit-learn, astropy, IPython, matplotlib, george, emcee, tqdm
 - PyTransit 2.0, PyDE

## Installation

    git clone git@github.com:hpparvi/MuSCAT2_transit_pipeline.git
    cd MuSCAT2_transit_pipeline
    python setup.py install

## Usage

### Photometry pipeline
#### Organisation

    m2organize raw_dir org_root

The organisation script `m2organize` takes the raw observation directory `raw_dir` and copies and organises the data in it in a directory created under an organised data root directory as `org_root/raw_dir`. In general, `raw_dir` is of format `yymmdd`, and the organised directory will then be `org_root/yymmdd`.

#### Astrometry

    m2astrometry -n n_processes

The astrometry script needs to be run from the organised data directory and takes the number of parallel processes as its main (optional) argument. This should be no larger than the number of cores available.

#### Photometry

    m2photometry XX XX

### Transit Analysis

    m2fit input_file

## Collaborators

- Instituto de Astrofísica de Canarias
- University of Tokyo
- National Astronomical Observatory of Japan
- The Graduate University for Advanced Studies
- Astrobiology Center

## Contributors

- Hannu Parviainen

---

<center>
&copy; 2018 Hannu Parviainen
</center>
