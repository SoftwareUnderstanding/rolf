**Note: The Legacy Survey of Space and Time Dark Energy Science Collaboration (LSST-DESC) is responsible for ongoing development of `qp` in [a separate repository](https://github.com/LSSTDESC/qp).
This repository contains the code used in [Malz, et. al. 2018. AJ. 156 1 35.](https://doi.org/10.3847/1538-3881/aac6b5) for the sake of reproducibility *(and if you want to reconstruct PDFs from the quantile parameterization, which is not yet implemented in the LSST-DESC version)*.**

---

# qp

Quantile parametrization for probability distribution functions.

[![Build Status](https://travis-ci.org/aimalz/qp.svg?branch=master)](https://travis-ci.org/aimalz/qp)[![Documentation Status](http://readthedocs.org/projects/qp/badge/?version=latest)](http://qp.readthedocs.io/en/latest/?badge=latest)


## Motivation
This repository exists for two reasons.

1. To be the home of `qp`, a python package for handling probability distributions using various parametrizations, including a set of quantiles;
2. To help us learn best practices in software carpentry.

## Examples

You can run the `qp` IPython notebooks live over the web at [Binder](http://mybinder.org):

[![Binder](http://mybinder.org/badge.svg)](http://mybinder.org/repo/aimalz/qp)

To make sure you are running the latest version, you can rebuild the Binder `qp` installation [here](http://mybinder.org/status/aimalz/qp).

You can also browse the (un-executed) notebooks here in this repo:

* **[Basic  Demo](http://htmlpreview.github.io/?https://github.com/aimalz/qp/blob/html/demo.html)** [(raw notebook)](https://github.com/aimalz/qp/blob/master/docs/notebooks/demo.ipynb)
* **[KL Divergence  Illustration](http://htmlpreview.github.io/?https://github.com/aimalz/qp/blob/html/kld.html)** [(raw notebook)](https://github.com/aimalz/qp/blob/master/docs/notebooks/kld.ipynb)

Also: [Read the Docs](http://qp.readthedocs.io/)


## People

* [Alex Malz](https://github.com/aimalz/qp/issues/new?body=@aimalz) (NYU)
* [Phil Marshall](https://github.com/aimalz/qp/issues/new?body=@drphilmarshall) (SLAC)

## License, Contributing etc

The code in this repo is available for re-use under the MIT license, which means that you can do whatever you like with it, just don't blame us. If you end up using any of the code or ideas you find here in your academic research, please cite us as `[Malz, et. al. 2018. AJ. 156 1 35.](https://doi.org/10.3847/1538-3881/aac6b5)\footnote{\texttt{https://github.com/aimalz/qp}}`. If you are interested in this project, please do drop us a line via the hyperlinked contact names above, or by [writing us an issue](https://github.com/aimalz/qp/issues/new). To get started contributing to the `qp` project, just fork the repo - pull requests are always welcome!
