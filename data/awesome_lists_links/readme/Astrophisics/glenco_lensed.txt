Lensed
======

[![Build Status](https://travis-ci.org/glenco/lensed.svg?branch=master)](https://travis-ci.org/glenco/lensed)
[![Documentation Status](http://readthedocs.org/projects/lensed/badge/?version=latest)](http://lensed.readthedocs.org/en/latest/?badge=latest)

Reconstruct gravitational lenses and lensed sources from strong lensing
observations.

Information regarding Lensed can be found on the [website] and [project] on
GitHub. The main [documentation] contains detailed instructions on how to set
up, run, configure and extend the code.

This project aims to be a community effort, and we are happy for any and all
contributions. The easiest part to play is to open [issues] whenever something
does not work, and create [pull requests] whenever you have fixed something
that was broken. Contributions in the form of new models are most welcome if
they are generally useful.


Installation
------------

Pre-compiled binary distributions of Lensed are available for the official
release versions, found on [GitHub](https://github.com/glenco/lensed/releases).
More information can be found in the [documentation](docs/releases.md).

There are a number of ways to get source distributions of the code, please see
the [documentation](docs/building.md).


Building
--------

Lensed is a simple Makefile project. If all dependencies are satisfied and
installed into the system, it should be possible to compile Lensed by simply
calling

    $ make

from the root of the source directory. For further information, please refer to
the [documentation](docs/building.md).


Usage
-----

The usual invocation of Lensed is with a [configuration](docs/configuration.md)
file which contains program options and the model that is being reconstructed.
The reconstruction is started using

    $ lensed config.ini

where `config.ini` is the name of the configuration file.

It is possible to split the configuration into different files, for example to
modularise options and individual aspects of the model. The following command
would run Lensed with the configuration taken from three different files:

    $ lensed options.ini lens.ini sources.ini

Repeated options are overwritten by the later configuration files.

Finally, it is possible to give any of the options (but not the model) directly
on the command line. For example, in order to change the number of live points
for a quick look at results, one could invoke Lensed in the following way:

    $ lensed config.ini --nlive=50

As before, the order of the given configuration files and options is important.

If Lensed is [built with XPA support](docs/building.md) or taken from a binary
release, direct updating of the reconstruction progress in SAOImage DS9 is
available.

    $ lensed config.ini --ds9

The `--ds9` option enables basic DS9 integration in Lensed. The flag can be set
from a [configuration file](docs/configuration.md) as well.


License
-------

The Lensed source code can be freely modified and distributed under the terms
of the [MIT license](LICENSE.txt). The authors also kindly ask that any and all
scientific work which benefits from this software cites the relevant articles.

This software makes use of the following libraries from third parties:

-   MultiNest, copyrighted by Farhan Feroz and Mike Hobson, with some
    limitations (especially for commercial use)
-   CFITSIO

Please see the [licenses file](docs/licenses.md) for copyright information.


Versions
--------

The Lensed project uses a form of [semantic versioning](http://semver.org).

Given a version number MAJOR.MINOR.PATCH, Lensed increments the

1.  MAJOR version for changes that render input files incompatible,
2.  MINOR version for changes that leave input files compatible but might
    lead to changes in results, and
3.  PATCH version for bugfix changes that do not alter results, unless the
    results are affected by the bugs fixed.

By comparing the output of `lensed --version` before installing a different
version of the code, the user can anticipate whether updates could possibly
break existing models or alter the results of a reconstruction.


[website]: http://glenco.github.io/lensed/
[project]: https://github.com/glenco/lensed
[documentation]: http://lensed.readthedocs.org
[issues]: https://github.com/glenco/lensed/issues
[pull requests]: https://github.com/glenco/lensed/pulls
