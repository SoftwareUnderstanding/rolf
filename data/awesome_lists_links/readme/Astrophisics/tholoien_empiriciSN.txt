# empiriciSN

empiriciSN is a software module for generating realistic supernova parameters given photometric observations of a potential host galaxy, based entirely on empirical correlations measured from supernova datasets. This code is intended to be used to improve supernova simulation for DES and LSST. It is extendable such that additional datasets may be added in the future to improve the fitting algorithm or so that additional light curve parameters or supernova types may be fit.

[![Build Status](https://travis-ci.org/tholoien/empiriciSN.svg?branch=master)](https://travis-ci.org/tholoien/empiriciSN)
[![DOI](https://zenodo.org/badge/61058789.svg)](https://zenodo.org/badge/latestdoi/61058789)

### SN Parameters
The code currently supports the generation of SALT2 parameters (stretch, color, and magnitude) for Type Ia supernovae.

### Host Parameters
Currently the code is trained based on the following host galaxy parameters:
* *ugriz* magnitudes and colors
* Separation of SN from host nucleus (angular and physical)
* Local surface brightness in *ugriz* bands, based on exponential or de Vaucouleurs profile fit

These same parameters are used to generate SN parameters for a given host. Photometry is K-corrected and corrected for Galactic extinction prior to correlations being calculated and SN properties being fit. 

### SN Datasets
The software has been trained using the following datasets:
* SNLS ([Guy et al. (2010)](http://cdsads.u-strasbg.fr/cgi-bin/nph-bib_query?2010A%26A...523A...7G&db_key=AST&nosetcookie=1), [Sullivan et al. (2010)](http://cdsads.u-strasbg.fr/abs/2011yCat..74060782S); 277 SNe Ia with SALT2 params, 231 with host params)
* SDSS ([Sako et al. (2104)](http://arxiv.org/abs/1401.3317); ~1400 SNe Ia)

## Using the code 

You will need Tom Holoien's `XDGMM` package, and its dependencies: 
```
pip install -r requirements.txt
pip install git+git://github.com/tholoien/XDGMM.git#egg=xdgmm
python setup.py install
```

Then see **[the demo notebook](https://github.com/tholoien/empiriciSN/blob/master/Notebooks/Demo.ipynb)** for a worked example `empiricSN` analysis.


## Contact

This is research in progress. All content is Copyright 2016 The Authors, and our code will be available for re-use under the MIT License (which basically means you can do anything you like with it but you can't blame us if it doesn't work). If you end up using any of the ideas or code in this repository in your own research, please cite [Holoien, Marshall, & Wechsler (2017)](http://adsabs.harvard.edu/abs/2016arXiv161100363H), and provide a link to this repo's URL: **https://github.com/tholoien/empiriciSN**. However, long before you get to that point, we'd love it if you got in touch with us! You can write to us with comments or questions any time using [this repo's issues](https://github.com/tholoien/empiriciSN/issues). We welcome new collaborators!

People working on this project:

* Tom Holoien (Ohio State, [@tholoien](https://github.com/tholoien/empiriciSN/issues/new?body=@tholoien))
* Phil Marshall (KIPAC, [@drphilmarshall](https://github.com/tholoien/empiriciSN/issues/new?body=@drphilmarshall))
* Risa Wechsler (KIPAC, [@rhw](https://github.com/tholoien/empiriciSN/issues/new?body=@rhw))

