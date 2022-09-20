# cosmic-kite
<img src="cosmic-kite2.jpg" style="float: left;" alt="drawing" width="180"/> *Dedicated with love and admiration to the memory of Diego Armando Maradona*

Cosmic-kite is a python software for the fast estimation of the TT Cosmic Microwave Background (CMB) power spectra corresponding to a set of cosmological parameters (using the function ```pars2ps```) or to estimate the maximum-likelihood cosmological parameters from a power spectra. (using the function ```ps2pars```).

This software is an auto-encoder that was trained and calibrated using power spectra from random cosmologies computed with the CAMB code. Fore more details please read https://arxiv.org/abs/2202.05853

# Installation

Cosmic-kite can be easylly installed in a python enviroment by doing:

```pip install git+https://github.com/Martindelosrios/cosmic-kite```

# Dependencies

As cosmic-kite is an auto-encoder that was previously trained using other python softwares, in order to use it, you will first need to install the following dependencies:

* ```tensorflow```
* ```numpy```
* ```scikit-learn==0.22.2.post1 ```

# Basic Usage
Let's estimate the power spectrum that correspond to the cosmological parameters estimated by Planck Collaboration.

```
from cosmic_kite import cosmic_kite

H0_true  = 67.32117
omb_true = 0.0223828
omc_true = 0.1201075
n_true   = 0.9660499
tau_true = 0.05430842
As_true  = 2.100549e-9

true_pars = np.array([omb_true, omc_true, H0_true, n_true, tau_true, As_true]).reshape(1,-1)

# The input of the pars2ps function must be an array of shape (n, 6) 
#  where n is the number of cosmological models to be computed

ps = cosmic_kite.pars2ps(true_pars)[0]
```

Now, let's estimate the maximum-likelihood parameters from a giver power spectra

```
# The input of the ps2pars function must be an array of shape (n, 2450)
#  where n is the number of cosmological models to be computed
pred_pars = cosmic_kite.ps2pars(ps.reshape(1,-1))[0]

```

Fore more examples please go to https://github.com/Martindelosrios/cosmic-kite/tree/main/Examples.

If you want to reproduce the figures of 2202.05853 please go to https://github.com/Martindelosrios/cosmic-kite/tree/main/Examples/Figures.ipynb.

# Citation

If you use this code please cite 2202.05853 and refer to  this website.

# Authors

Martín de los Rios. Posdoctoral Resarcher @IFT/UAM.  <a itemprop="sameAs"  href="https://orcid.org/0000-0003-2190-2196" target="orcid.widget" rel="noopener noreferrer" style="vertical-align:top;"> <img src="https://orcid.org/sites/default/files/images/orcid_16x16.png" style="width:1em;margin-right:.5em;" alt="ORCID iD icon"></a>
