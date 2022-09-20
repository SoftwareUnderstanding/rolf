# uvplot

[![image](https://github.com/mtazzari/uvplot/actions/workflows/tests.yml/badge.svg)](https://github.com/mtazzari/uvplot/actions/workflows/tests.yml)
[![image](https://img.shields.io/pypi/v/uvplot.svg)](https://pypi.python.org/pypi/uvplot)
[![image](https://img.shields.io/github/release/mtazzari/uvplot/all.svg)](https://github.com/mtazzari/uvplot/releases)
[![image](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)
[![image](https://zenodo.org/badge/105298533.svg)](https://zenodo.org/badge/latestdoi/105298533)

A simple Python package to make nice plots of deprojected interferometric
visibilities, often called **uvplots** (see an example below).

**uvplot** also [makes it easy](https://uvplot.readthedocs.io/en/latest/basic_usage.html#exporting-visibilities-from-ms-table-to-uvtable-ascii)
to export visibilities from
`MeasurementSets` to `uvtables`, a handy format for fitting the data
(e.g., using [Galario](https://github.com/mtazzari/galario)).

**uvplot** can be used as a standalone Python package (available on 
[PyPI](https://pypi.python.org/pypi/uvplot)) and also inside [NRAO CASA](https://casa.nrao.edu/) 6.x. 
It can be installed in a Python environment and in a CASA terminal with:
```bash
pip install uvplot
```

An example uvplot made with [this simple code](https://uvplot.readthedocs.io/en/latest/basic_usage.html#plotting-visibilities):
<p align="center">
    <img src="docs/images/uvplot.png" width="600" height="600">
</p>

If you are interested, have feature requests, or encounter issues,
consider creating an [Issue](https://github.com/mtazzari/uvplot/issues)
or writing me an [email](mtazzari@ast.cam.ac.uk). I am happy to have your
feedback!

Check out the [documentation](https://uvplot.readthedocs.io/) and
the [installation instructions](https://uvplot.readthedocs.io/en/latest/install.html).

**uvplot** is used in a growing number of publications; 
[at this page](https://ui.adsabs.harvard.edu/#search/q=citations(bibcode%3A2017zndo...1003113T)%20&sort=date%20desc%2C%20bibcode%20desc&p_=0) you can find an
updated list.

If you use **uvplot** for your publication, please cite the [Zenodo
reference](https://zenodo.org/badge/latestdoi/105298533):

    @software{uvplot_tazzari,
      author       = {Marco Tazzari},
      title        = {mtazzari/uvplot},
      month        = oct,
      year         = 2017,
      publisher    = {Zenodo},
      doi          = {10.5281/zenodo.1003113},
      url          = {https://doi.org/10.5281/zenodo.1003113}
    }

## License

**uvplot** is free software licensed under the LGPLv3 License. For more
details see the LICENSE.

© Copyright 2017-2021 Marco Tazzari and contributors.
