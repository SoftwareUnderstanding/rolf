![Harmonia](https://github.com/MikeSWang/Harmonia/raw/master/docs/source/_static/Harmonia.png)

[![arXiv eprint](
https://img.shields.io/badge/arXiv-2007.14962-important
)](https://arxiv.org/abs/2007.14962)
[![GitHub release (latest by date)](
https://img.shields.io/github/v/release/MikeSWang/Harmonia?label=release
)](https://github.com/MikeSWang/Harmonia/releases/latest)
[![Documentation status](
https://readthedocs.org/projects/harmonia/badge/?version=latest
)](https://harmonia.readthedocs.io/en/latest)
[![Build status](
https://travis-ci.com/MikeSWang/Harmonia.svg?branch=master
)](https://travis-ci.com/MikeSWang/Harmonia)
[![Licence](
https://img.shields.io/badge/licence-GPLv3-informational
)](https://github.com/mikeswang/Harmonia/tree/master/LICENCE)


# Hybrid-Basis Inference for Large-Scale Galaxy Clustering

<span style="font-variant: small-caps">Harmonia</span> is a Python package
that combines clustering statistics decomposed in spherical and Cartesian
Fourier bases for large-scale galaxy clustering likelihood analysis.


## Installation

We recommend that you first install
[``nbodykit``](https://nbodykit.readthedocs.io/en/latest/) by following
these [instructions](
https://nbodykit.readthedocs.io/en/latest/getting-started/install.html).

After that, you can install
<span style="font-variant: small-caps">Harmonia</span> simply using ``pip``:

```bash
pip install harmoniacosmo
```

Note that only here does the name "harmoniacosmo" appear because
unfortunately on PyPI the project name "harmonia" has already been taken.


## Documentation

API documentation can be found at [mikeswang.github.io/Harmonia](
https://mikeswang.github.io/Harmonia).  Tutorials (in the format of
notebooks) will be gradually added in the future; for now, scripts in
[``application/``](
https://github.com/MikeSWang/Harmonia/tree/master/application) may offer
illustrative examples of how to use
<span style="font-variant: small-caps">Harmonia</span>.


## Attribution

If you would like to acknowledge this work, please cite
[Wang et al. (2020)](https://arxiv.org/abs/2007.14962). You may use the
following BibTeX record.

    @article{Wang_2020b,
        author={Wang, M.~S. and Avila, S. and Bianchi, D. and Crittenden, R. and Percival, W.~J.},
        title={Hybrid-basis inference for large-scale galaxy clustering: combining spherical and {Cartesian} {Fourier} analyses},
        year={2020},
        eprint={2007.14962},
        archivePrefix={arXiv},
        primaryClass={astro-ph.CO},
    }


## Licence

Copyright 2020, M S Wang

<span style="font-variant: small-caps">Harmonia</span> is made freely
available under the
[GPL v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html) licence.
