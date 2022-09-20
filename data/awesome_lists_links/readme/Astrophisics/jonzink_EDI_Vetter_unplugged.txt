<img src="http://www.jonzink.com/images/ediWhite3.png">

This software identifies false positive transit signals using [TLS](https://github.com/hippke/tls) information and has been simplified from the full [EDI-Vetter](https://github.com/jonzink/EDI-Vetter) algorithm for easy implementation with the [TLS](https://github.com/hippke/tls) output.

<a href="https://zenodo.org/badge/latestdoi/200920137"><img src="https://zenodo.org/badge/200920137.svg" alt="DOI"></a>   

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for research, development, and testing purposes. EDI-Vetter Unplugged was written in Python 3.4 

### Prerequisites

Several python packages are required to run this software. Here are a few:  TLS, Numpy, scipy, and astropy

EDI-Vetter Unplugged is meant to utilize the output provided by the [TLS](https://github.com/hippke/tls) software package. We remind users to cite both packages appropriately.  


### Installation

EDI-Vetter Unplugged can now be easily installed via pip

```
$ pip install EDIunplugged
```

## Running EDI-Vetter Unplugged in Python

Here we provide a quick example.

Begin by importing the necessary packages in to Python
```
>>> import EDIunplugged as EDI
>>> import transitleastsquares
```
Run the light curve file through TLS
```
>>> model = transitleastsquares(Time, Flux)
>>> tlsOut = model.power()
```
Now you can set up the EDI-Vetter Unplugged parameters object with the TLS output object. For a quick-start you can enter:
```
>>> params=EDI.parameters(tlsOut)
```
For a more detailed analysis, you can provide additional information about your search:
```
>>> params=EDI.parameters(tlsOut, limbDark=[0.48, 0.19], impact=0, snrThreshold=7, minTransit=3)
```
Here you have the option to provide the quadratic limb darkening values, the transit impact parameter, the desired SNR threshold, and/or the minimum number of transits considered for a valid detection. The default values have been listed in the example above.

Now you can run all of the vetting metrics on the signal
```
>>> params=EDI.Go(params, print=True)
```
Once completed, EDI-Vetter Unplugged will print out a vetting report (if "print" is set to True):
```
 ___________ _____      _   _      _   _            
|  ___|  _  \_   _|    | | | |    | | | |           
| |__ | | | | | |______| | | | ___| |_| |_ ___ _ __ 
|  __|| | | | | |______| | | |/ _ \ __| __/ _ \ '__|
| |___| |/ / _| |_     \ \_/ /  __/ |_| ||  __/ |   
\____/|___/  \___/      \___/ \___|\__|\__\___|_|   Unplugged
   
==========================================
            Vetting Report
==========================================
        Flux Contamination : False
	 Too Many Outliers : False
  Too Many Transits Masked : True
Odd/Even Transit Variation : False
      Signal is Not Unique : True
   Secondary Eclipse Found : False
Low Transit Phase Coverage : False
Transit Duration Too Long : False
==========================================
Signal is a False Positive : True
```
In this case, the signal was not unique within the light curve and is likely a false positive. Additionally, the number of meaningful transits fell below the desired threshold.

| Output | Description |
| --- | --- |
| `fluxContamFP` | Was neighboring flux contamination contributing significantly? |
| `outlierTranFP` | Was there an abundance of model outliers, indicating a systematic issue?  |
| `transMaskFP` | Were the individual transits questionable?  |
| `evenOddFP` | Does the signal deviate significantly between even and odd transits? |
| `uniqueFP` | Does the signal appear similar to other signals within the light curve? |
| `secEclipseFP` | Does the signal appear to have a secondary eclipse? |
| `phaseCoverFP` | Does the signal lack sufficient data to detect a meaningful transit? |
| `tranDurFP` | Is the transit duration too long when compared to the period? |
| `FalsePositive` | Overall, does the signal appear to be a false positive? |


 You can access the suggested classification from EDI-Vetter Unplugged using the "params" output object:
```
>>> print(params.FalsePositive)
True
>>> print(params.fluxContamFP)
False
```
Alternatively, you can enter information about a potential contaminating star by indicating the photometric aperture size in pixels ("photoAp"), the telescope collected from ("telescope"), the separation in arcseconds from target star and the contaminating source ("deltaDist"), and the difference in visual magnitude between the sources ("deltaMag"; i.e., secondary source magnitude - primary source magnitude ). Note: EDI-Vetter Unplugged is currently only applicable with "Kepler", "K2", and "TESS" telescope choices.

```
>>> params=EDI.Go(params, deltaMag=10, deltaDist=1000, photoAp=25, telescope="TESS")
```
It is important to note this is not the Full EDI-Vetter suite of vetting metrics, but rather a large fraction that could be easily implemented alongside TLS. Thus, EDI-Vetter Unplugged is likely to have a higher completeness, but a lower reliability when compared to the original algorithm. 

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

```
