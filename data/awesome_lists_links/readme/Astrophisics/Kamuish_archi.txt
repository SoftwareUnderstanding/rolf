[![Documentation Status](https://readthedocs.org/projects/archi/badge/?version=latest)](https://archi.readthedocs.io/en/latest/?badge=latest)  [![PyPI version fury.io](https://badge.fury.io/py/pyarchi.svg)](https://pypi.org/project/pyarchi/) [![PyPI pyversions](https://img.shields.io/pypi/pyversions/pyarchi.svg)](https://pypi.org/project/pyarchi/) [![DOI:10.1093/mnras/staa1443](https://zenodo.org/badge/DOI/10.1007/978-3-319-76207-4_15.svg)](https://doi.org/10.1093/mnras/staa1443)

# ARCHI - An expansion to the CHEOPS mission official pipeline


High precision time-series photometry from space is being used for a number of scientific cases. In this context, the recently launched CHaracterizing ExOPlanet Satellite (CHEOPS) (ESA) mission promises to bring 20 ppm precision over an exposure time of 6 h, when targeting nearby bright stars, having in mind the detailed characterization of exoplanetary systems through transit measurements. However, the official CHEOPS (ESA) mission pipeline only provides photometry for the main target (the central star in the field). In order to explore the potential of CHEOPS photometry for all stars in the field,  we present archi, an additional open-source pipeline module to analyse the background stars present in the image. As archi uses the official data reduction pipeline data as input, it is not meant to be used as an independent tool to process raw CHEOPS data but, instead, to be used as an add-on to the official pipeline. We test archi using CHEOPS simulated images, and show that photometry of background stars in CHEOPS images is only slightly degraded (by a factor of 2–3) with respect to the main target. This opens a potential for the use of CHEOPS to produce photometric time-series of several close-by targets at once, as well as to use different stars in the image to calibrate systematic errors. We also show one clear scientific application where the study of the companion light curve can be important for the understanding of the contamination on the main target.

# ARCHI - a quick preview 

Here we have the masks used for the analysis of a simulated data set, for each individual image:

![Alt Text](https://github.com/Kamuish/archi/blob/master/docs/archi_info/star_tracking.gif)


# How to install archi 

The pipeline is written in Python3, and most features should work on all versions. However, so far, it was only tested on python 3.6, 3.7 and 3.8

To install, simply do :

    pip install pyarchi 

To see bug fixes and the new functionalities of each version refer to the [official documentation](https://archi.readthedocs.io/en/latest/archi_info/release.html)

# How to use the library 

A proper introduction to the library, alongside documentation of the multiple functions and interfaces can be found [here](https://archi.readthedocs.io/en/latest/). 

If you use the pipeline, cite the article 

    @article{Silva_2020,
       title={ARCHI: pipeline for light curve extraction of CHEOPS background stars},
       ISSN={1365-2966},
       url={http://dx.doi.org/10.1093/mnras/staa1443},
       DOI={10.1093/mnras/staa1443},
       journal={Monthly Notices of the Royal Astronomical Society},
       publisher={Oxford University Press (OUP)},
       author={Silva, André M and Sousa, Sérgio G and Santos, Nuno and Demangeon, Olivier D S and Silva, Pedro and Hoyer, S and Guterman, P and Deleuil, Magali and Ehrenreich, David},
       year={2020},
       month={May}
    }

# Known Problems


 [1] There is no correction for cross-contamination between stars
 
 [2] If we have data in the entire 200*200 region (not expected to happen) and using the "dynam" mask for the background stars it might "hit" one of the edges of the image. In such case, larger masks will not increase in the direction in which the edge is reached. However, the mask can still grow towards the other directions, leading to masks significantly larger than the original star. In such cases, we recommend to manually change the mask size on the "optimized factors" file.
