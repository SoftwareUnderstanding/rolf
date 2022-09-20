# EDI-Vetter
This is a program meant identify false positive transit signal in the K2 data set. This program has been simplified to test single transiting planet signals. Systems with multiple signals require additional testing, which will be made available in a later iteration.

UPDATE: There has been a signficant desire for a [TLS](https://github.com/hippke/tls) based version of EDI-Vetter. Thus, I present [EDI-Vetter Unplugged](https://github.com/jonzink/EDI_Vetter_unplugged), an easily impletemeneted suite of vetting metrics built to run alongside [TLS](https://github.com/hippke/tls).

<a href="https://zenodo.org/badge/latestdoi/200920137"><img src="https://zenodo.org/badge/200920137.svg" alt="DOI"></a>   

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for research, development, and testing purposes. EDI-Vetter was written in Python 3.4 

### Prerequisites

Several python packages are required to run this software. Here are a few: Pandas, Numpy, emcee, scipy, lmfit, batman, astropy

EDI-Vetter currently relies on several features provided by the [Terra](https://github.com/petigura/terra) software package. We have included a copy in this repositories, but remind users to cite appropriately.  




## Running EDI-Vetter in Python

Here we provide a quick example using the light curve of K2-138.

Begin by opening Python in the appropriate directory. 
```
$ python
```
Now import the necessary packages
```
>>> import pandas as pd
>>> import EDI_Vetter
```
Import the light curve file
```
>>> lc=pd.read_csv("K2_138.csv")
```
Now you can set up the EDI-Vetter parameters object with the appropriate transit signal parameters 
```
>>> params=EDI_Vetter.parameters(per=8.26144,  t0=2907.6451,  radRatio=0.0349,  tdur=0.128,  lc=lc)
```
It is essential that EDI-Vetter re-fits the light curve to measure changes from the transit detection.
```
>>> params=EDI_Vetter.MCfit(params)
```
Now you can run all of the vetting metrics on the signal
```
>>> params=EDI_Vetter.Go(params,delta_mag=10,delta_dist=1000, photoAp=41)
```

## Attribution
Please cite as [Zink et al. (2020a)](https://ui.adsabs.harvard.edu/abs/2020AJ....159..154Z/abstract).
```

@ARTICLE{2020AJ....159..154Z,
       author = {{Zink}, Jon K. and {Hardegree-Ullman}, Kevin K. and
         {Christiansen}, Jessie L. and {Dressing}, Courtney D. and
         {Crossfield}, Ian J.~M. and {Petigura}, Erik A. and
         {Schlieder}, Joshua E. and {Ciardi}, David R.},
        title = "{Scaling K2. II. Assembly of a Fully Automated C5 Planet Candidate Catalog Using EDI-Vetter}",
      journal = {\aj},
     keywords = {Astrophysics - Earth and Planetary Astrophysics, Astrophysics - Instrumentation and Methods for Astrophysics},
         year = 2020,
        month = apr,
       volume = {159},
       number = {4},
          eid = {154},
        pages = {154},
          doi = {10.3847/1538-3881/ab7448},
archivePrefix = {arXiv},
       eprint = {2001.11515},
 primaryClass = {astro-ph.EP},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2020AJ....159..154Z},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
