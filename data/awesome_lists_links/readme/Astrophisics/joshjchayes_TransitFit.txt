
# TransitFit

**Author** - Joshua Hayes (joshjchayes@gmail.com), as part of the [SPEARNET collaboration](mailto:eamonn.kerins@manchester.ac.uk).

## Overview
[TransitFit](https://transitfit.readthedocs.io/en/latest/) is designed for exoplanetary transmission spectroscopy studies and offers a flexible approach to fitting single or multiple transits of an exoplanet at different observation wavelengths.  It possesses the functionality to efficiently couple host limb-darkening parameters to a range of physical models across different wavelengths, through the use of the [Limb darkening toolkit (ldtk)](https://github.com/hpparvi/ldtk) and the [Kipping parameterisations of two-parameter limb darkening models](https://arxiv.org/abs/1308.0009). TransitFit uses [batman](https://www.cfa.harvard.edu/~lkreidberg/batman/index.html) to handle transit light curve modelling, and sampling and retrieval uses the nested sampling algorithm available through [dynesty](https://dynesty.readthedocs.io/en/latest/index.html).

<a name="installation"></a>
## Installation

Please note that TransitFit currently only runs on UNIX-based machines.

Along with an installation of Python 3 (with the standard Conda distribution packages), TransitFit requires the following packages to be installed:

- [dynesty](https://dynesty.readthedocs.io/en/latest/index.html)

- [batman](https://www.cfa.harvard.edu/~lkreidberg/batman/index.html)

- [Limb darkening toolkit (ldtk)](https://github.com/hpparvi/ldtk)


<a name="guide"></a>
## User guide
The documentation for TransitFit can be found [here](https://transitfit.readthedocs.io/en/latest/)

<a name="citing"></a>
## Citing TransitFit
If you have used TransitFit in your work, please cite the [accompanying paper](https://ui.adsabs.harvard.edu/abs/2021arXiv210312139H/abstract). If you are using BibTeX, you can add the following to your .bib file:

```
@ARTICLE{2021arXiv210312139H,
       author = {{Hayes}, J.~J.~C. and {Kerins}, E. and {Morgan}, J.~S. and {Humpage}, A. and {Awiphan}, S. and {Charles-Mindoza}, S. and {McDonald}, I. and {Inyanya}, T. and {Padjaroen}, T. and {Munsaket}, P. and {Chuanraksasat}, P. and {Komonjinda}, S. and {Kittara}, P. and {Dhillon}, V.~S. and {Marsh}, T.~R. and {Reichart}, D.~E. and {Poshyachinda}, S.},
        title = "{TransitFit: an exoplanet transit fitting package for multi-telescope datasets and its application to WASP-127~b, WASP-91~b, and WASP-126~b}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics - Earth and Planetary Astrophysics, Astrophysics - Instrumentation and Methods for Astrophysics},
         year = 2021,
        month = mar,
          eid = {arXiv:2103.12139},
        pages = {arXiv:2103.12139},
archivePrefix = {arXiv},
       eprint = {2103.12139},
 primaryClass = {astro-ph.EP},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2021arXiv210312139H},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
