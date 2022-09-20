# deepSIP

[![Build Status](https://travis-ci.org/benstahl92/deepSIP.svg?branch=master)](https://travis-ci.org/benstahl92/deepSIP) [![Documentation Status](https://readthedocs.org/projects/deepsip/badge/?version=latest)](https://deepsip.readthedocs.io/en/latest/?badge=latest)
 [![codecov](https://codecov.io/gh/benstahl92/deepSIP/branch/master/graph/badge.svg)](https://codecov.io/gh/benstahl92/deepSIP) [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) [![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)

deepSIP (deep learning of Supernova Ia Parameters) is an open-source toolkit for measuring the phase and light-curve shape (parameterized by SNooPy’s &Delta;*m*<sub>15</sub>) of a Type Ia Supernova (SN Ia) from an optical spectrum. The primary contents of the package are a set of three trained Convolutional Neural Networks (CNNs) for the aforementioned purposes, but tools for preprocessing spectra, modifying the neural architecture, training models, and sweeping through hyperparameters are also included.

If you use deepSIP in your research, please cite the following paper: [Stahl et al. (2020)](https://ui.adsabs.harvard.edu/abs/2020MNRAS.496.3553S/abstract)

## Installation

First, you’ll need to clone deepSIP and enter its directory.

```bash
git clone https://github.com/benstahl92/deepSIP.git
cd deepSIP
```

(optional) It is recommended that you use a virtual environment for deepSIP and its dependencies.

```bash
python -m venv dsenv # create virtual environment (one time only)
source dsenv/bin/activate # each time you need to activate the environment
deactivate # if/when you need to leave the environment
```

Install dependencies and deepSIP.

```bash
pip install -r requirements.txt
pip install .
```

## Standard Usage

```python
from deepSIP import deepSIP
ds = deepSIP()
# spectra is a pd.DataFrame with columns including ['SN', 'filename', 'z']
predictions = ds.predict(spectra, threshold = 0.5, status = True)
```

Full documentation is available on [readthedocs](https://deepSIP.readthedocs.io/en/latest/?badge=latest#).

## Contributing

We welcome community involvement in the form of bug fixes, architecture improvements and new trained models, additional high-quality data (spectra and photometry), and expanded functionality. To those wishing to participate, we ask that you fork the this repository and issue a pull request with your changes (along with unit tests and a description of your contribution).

deepSIP is developed and maintained by Benjamin Stahl under the supervision of Prof. Alex Filippenko at UC Berkeley.
