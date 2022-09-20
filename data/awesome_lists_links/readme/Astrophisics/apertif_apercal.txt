# About

Apercal - the APERTIF calibration and imaging pipeline.

https://www.astron.nl/astronomy-group/apertif/apertif

https://github.com/apertif/apercal/wiki/pipeline-overview

# Installation

Get `getdata_alta` package from https://github.com/cosmicpudding/getdata_alta and add it to your PYTHONPATH 
or add getdata_alta.py to your python-package directory.

```bash
$ pip install .
```

Currently only works with Python 2.

# CWL

This pipeline has been partially encapsulated using Docker and CWL. The getdata, preflag crosscal, conver, scal and
continuum steps should mostly work, but probably require adjusting for recent modifications to the source code.
To communicate with the ALTA for the getdata step proper credentials are required.

Requirements:

 * Docker
 * Python 3
 
The Makefile in the root of this project contains various helper targets showing how to run the pipeline and how to
set it up. If you run `make run` a Python virtual environment will be initiated and the pipeline will run with the
test data. Note that the pipeline is not idempotent, after running the pipeline the test data need to be restored
(`make clean`).



# Development
[![Build Status](https://travis-ci.org/apertif/apercal.svg?branch=master)](https://travis-ci.org/apertif/apercal)
