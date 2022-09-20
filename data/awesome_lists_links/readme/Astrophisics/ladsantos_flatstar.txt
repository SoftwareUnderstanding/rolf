# flatstar

`flatstar` is a pure-Python tool for drawing stellar disks with scientifically-rigorous limb darkening. Each pixel has an accurate fractional intensity in relation to the total stellar intensity of `1.0`. It is ideal for ray-tracing simulations of stars and planetary transits.

[![Build Status](https://travis-ci.com/ladsantos/flatstar.svg?branch=main)](https://travis-ci.com/ladsantos/flatstar) [![Coverage Status](https://coveralls.io/repos/github/ladsantos/flatstar/badge.svg?branch=main)](https://coveralls.io/github/ladsantos/flatstar?branch=main)
[![Documentation Status](https://readthedocs.org/projects/flatstar/badge/?version=latest)](https://flatstar.readthedocs.io/en/latest/?badge=latest)


![flatstar](assets/flatstar.png)

## Features

* Blazing fast! There are no for-loops involved, only `numpy` arrays manipulation and `pillow` Image objects.
* It has the most well-known limb-darkening laws: linear, [quadratic](https://ui.adsabs.harvard.edu/abs/1950HarCi.454....1K/abstract), [square-root](https://ui.adsabs.harvard.edu/abs/1992A%26A...259..227D/abstract), [logarithmic](https://ui.adsabs.harvard.edu/abs/1970AJ.....75..175K/abstract), [exponential](https://ui.adsabs.harvard.edu/abs/2003A%26A...412..241C/abstract), [Sing et al. (2009)](https://ui.adsabs.harvard.edu/abs/2009A%26A...505..891S/abstract), [Claret et al. (2000)](https://ui.adsabs.harvard.edu/abs/2000A%26A...363.1081C/abstract).
* You can implement your own custom limb-darkening law.
* Supersampling for the situations where you need coarse arrays but do not want to lose precision in stellar disk intensity (i.e., no hard pixel boundaries).
* Upscaling for the situations where you want to save on computation time but need high-resolution intensity maps (the price to pay here is that there is some precision loss in intensities).
* Resampling is done with the C-libraries of `pillow`, which means they are as fast as it goes.
* You can add planetary transits with pixel-perfect settings for impact parameter and first/fourth contacts.

## Installation

You can install `flatstar` either with `pip` or by compiling from source. Coming soon: you will also be able to install it using `conda-forge`. Any of these methods will also install the dependencies `numpy` and `pillow` if they are not yet installed.

### Using `pip` (more stable version)

Simply run the following line in your terminal and you are good to go:

```angular2html
pip install flatstar
```

### Compile from source (development version)

First clone this repository and navigate to it:

```angular2html
git clone https://github.com/ladsantos/flatstar.git && cd flatstar
```

And now install it:

```angular2html
python setup.py install
```