# Cosmic variance calculator
This is a quick and easy PYTHON cosmic variance calculator based on [Ucci et al. (2020)](https://arxiv.org/abs/2004.11096).

[![Open Source Love](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](https://opensource.org/licenses/Apache-2.0) [![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/grazianoucci/cosmic_variance/graphs/contributors)

## Prerequisites
To run the cosmic variance calculator script you need the following PYTHON packages: numpy, gzip and pickle.

## How to
We provide three simple functions that allow the user to compute the cosmic variance during the Epoch of Reionization (EoR) for the UV Luminosity Function (UV LF), Stellar Mass Function (SMF), and Halo Mass Function (HMF).

The user should pass to the functions the following data:
- redshift (allowed range: 6 - 12)
- redshift interval z_max - z_min (allowed range: 0.05 - 1)
- survey area [square arcmins] (allowed range: 1 - 1000)
- UV absolute magnitude (allowed range: -23 - -11.5)

  or
  Halo mass [log10 (solar masses)] (allowed range: 8 - 13)
  
  or
  Stellar mass [log10 (solar masses)] (allowed range: 5 - 12)
- model name: (allowed values: 'photoionization' (recommended), 'early_heating', 'jeans_mass')

The functions give as output the cosmic variance expressed in percentage.
We also provide simple examples, to get a clear idea of how to use the functions described above.

## Authors
* **Graziano Ucci** - *Kapteyn Astronomical Institute, Groningen* - [github](https://github.com/grazianoucci)

See also the list of [contributors](https://github.com/grazianoucci/cosmic_variance/contributors) who participated in this project.

## Citing
If you use this code for your work, please cite it with
```
@ARTICLE{2020arXiv200411096U,
       author = {{Ucci}, Graziano and {Dayal}, Pratika and {Hutter}, Anne and {Yepes}, Gustavo and {Gottl{\"o}ber}, Stefan and {Legrand}, Laurent and {Pentericci}, Laura and {Castellano}, Marco and {Choudhury}, Tirthankar Roy},
        title = "{Astraeus II: Quantifying the impact of cosmic variance during the Epoch of Reionization}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics - Astrophysics of Galaxies},
         year = 2020,
        month = apr,
          eid = {arXiv:2004.11096},
        pages = {arXiv:2004.11096},
archivePrefix = {arXiv},
       eprint = {2004.11096},
 primaryClass = {astro-ph.GA},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2020arXiv200411096U},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
