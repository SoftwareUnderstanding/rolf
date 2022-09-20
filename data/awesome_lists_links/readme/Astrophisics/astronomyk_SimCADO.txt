# SimCADO - the instrument data simulator for MICADO

[![Documentation Status](https://readthedocs.org/projects/simcado/badge/?version=latest)](https://simcado.readthedocs.io/en/latest/?badge=latest)
[![Testing Status](https://travis-ci.org/astronomyk/SimCADO.svg?branch=master)](https://travis-ci.org/astronomyk/SimCADO.svg?branch=master)


Documentation for SimCADO can be found here:

[https://readthedocs.org/projects/simcado/](https://readthedocs.org/projects/simcado/)

## SimCADO in a nutshell
SimCADO is a python package designed to simulate the effects of the Atmosphere, E-ELT, and MICADO instrument on incoming light. The current version (v0.2) can simulate the MICADO imaging modi (4mas and 1.5mas per pixel in the wavelength range 0.7µm to 2.5µm).

### Reference Material

* The inner workings of SimCADO are described in detail in [Leschinski et al. (2016)](https://arxiv.org/pdf/1609.01480v1.pdf)

* The current status of MICADO is described in [Davies et al. (2016)](https://arxiv.org/pdf/1607.01954.pdf)


## Downloading and Installing
For more information, see the the documentation

**SimCADO has only been tested in Python 3.x**. 

It is highly recommended to use Python 3, however the basics of generating images will still work in Python 2.7. We cannot guarantee this though. See the [Features](Features.md) page for more info on which functions with which Python version.

The quick way:

    $ pip install simcado

The **first time** in python 

    >>> import simcado
    >>> simcado.get_extras()
    >>>
    >>> # !! Only works in Python 3 - See Downloads section
    >>> simcado.install_noise_cube()
    
### Keeping SimCADO updated

As MICADO developes, the data files that SimCADO uses will also be updated. Therefore before you do any major work with SimCADO we *HIGHLY* recommend calling:

    >>> simcado.get_extras()


## Running a simulation in 3 lines

**the keyword OBS_EXPTIME has been replaced by OBS_DIT in the latest version of SimCADO**

The easiest way to run a simulation is to create, or load, a Source object and then call the `.run()` command. If you specify a filename, the resulting image will be output to a FITS file under that name. If you do not specify a filename, the output will be returned to the console/notebook as an `astropy.io.fits.HDUList` object.

To begin, we will import the simcado module (assuming it is already installed).

    >>> import simcado

At the very least, we need to create a `Source` object which contains both spatial and spectral information on our object of interest. Here we use the built-in command `.source.source_1E4_Msun_cluster()` to create a `Source`-object for a 10000-Msun stellar cluster. (See [Creating Sources](examples/Source.md) for more information).

    >>> src = simcado.source.source_1E4_Msun_cluster()

We now pass the `source` object through SimCADO. This is as easy as calling `.run()`. If we specify a `filename`, SimCADO will write the output to disk in the form of a FITS file. If no `filename` is given, then SimCADO returns an `astropy.io.fits` object to the console/notebook.

    >>> simcado.run(src, filename="my_first_sim.fits")

### Changing simulation parameters

The `sim.run()` also takes any [configuration keywords](Keywords.md) as parameters for running the simulation. For example, the default exposure time for the simulation is 60 seconds, however this can be increased of decreased by using the keyword `OBS_DIT` (and/or combining it with `OBS_NDIT`). A stacked 6x 10 minute observation sequence would look like:

    >>> simcado.run(src, filename="my_first_sim.fits", OBS_DIT=600, OBS_NDIT=6)
    
That's it. Of course SimCADO can also go in the other direction, providing many more levels of complexity, but for that the reader is directed to the examples pages and/or the [API](API/_build/index.html) documentation

### SimCADO building blocks
For a brief explanation of how SimCADO works and which classes are relevant, please see either the [Getting Started](GettingStarted.md) or [SimCADO in depth](deep_stuff/SimCADO.md) section.

## Bugs and Issues

We freely admit that there may still be several bugs that we have not found. If you come across an buggy part of SimCADO, *please please* tell us. We can't make SimCADO better if we don't know about things.

The preferable option is to open an issue on our Github page: [astronomyk/SimCADO/issues](https://github.com/astronomyk/SimCADO/issues), or you can contact either one of us directly.

Please always include the output of
 
    >>> simcado.bug_report()


## Contact

For questions and complaints alike, please contact the authors:

* [kieran.leschinski@univie.ac.at]()
* [oliver.czoske@univie.ac.at]()
* [miguel.verdugo@univie.ac.at]()

**Developers (Vienna):** Kieran Leschinski, Oliver Czoske, Miguel Verdugo

**Data Flow Team Leader (Gronigen):** Gijs Verdoes Kleijn

**MICADO home office (MPE):** http://www.mpe.mpg.de/ir/micado
