
Exoplanet Detection Map Calculator (Exo-DMC)
==========

Information
-----------
This repository includes the first version of the `Exo-DMC` (Exoplanet Detection Map Calculator), a Monte Carlo tool for the statistical analysis of exoplanet surveys results.

The tool combines the information on the target stars with the instrument detection limits to estimate the probability of detection of a given synthetic planet population, ultimately generating detection probability maps. 

Requirements
------------

This package relies on usual packages for data science and astronomy: [numpy](https://numpy.org/), [scipy](https://www.scipy.org/), [pandas](https://pandas.pydata.org/), [matplotlib](https://matplotlib.org/) and [astropy](https://www.astropy.org/).

# Installation: 
The easiest is to install `Exo-DMC` using `pip`:

```sh
pip install ExoDMC
```
                                                   
Otherwise your can download the current repository and install the package manually:

```sh
cd ExoDMC/
python setup.py install
```

Examples
--------

The package is not fully documented, but [examples](https://github.com/mbonav/Exo-DMC/tree/master/examples) are provided.

If you find a bug or want to suggest improvements, please create a [ticket](https://github.com/mbonav/Exo-DMC/issues)

Recent papers using the Exo-DMC: 
-----------------------
* `The JWST Early Release Science Program for the Direct Imaging & Spectroscopy of Exoplanetary Systems`[Hinkley et al. 2022arXiv220512972H](https://ui.adsabs.harvard.edu/abs/2022arXiv220512972H/abstract)
* `New binaries from the SHINE survey` [Bonavita et al. 2021arXiv210313706B](https://ui.adsabs.harvard.edu/abs/2021arXiv210313706B/abstract)
* `Large Adaptive Optics Survey for Substellar Objects around Young, Nearby, Low-mass Stars with Robo-AO`[Salama, M et al. 2021AJ....162..102S](https://ui.adsabs.harvard.edu/abs/2021AJ....162..102S/abstract)
* `The SPHERE infrared survey for exoplanets (SHINE)- I Sample definition and target characterization` [Desidera, S. et al. 2021A&A...651A..70D](https://ui.adsabs.harvard.edu/abs/2021A%26A...651A..70D/abstract)
* `The HOSTS survey: evidence for an extended dust disk and constraints on the presence of giant planets in the Habitable Zone of β Leo` [Defrère, D. et al. 2021AJ....161..186D](https://ui.adsabs.harvard.edu/abs/2021AJ....161..186D/abstract) 
* `Direct imaging of sub-Jupiter mass exoplanets with James Webb Space Telescope coronagraphy` [Carter, et al. 2021, MNRAS, 501, 1999](https://arxiv.org/abs/2011.07075) 
* `Limits on the presence of planets in systems with debris disks: HD 92945 and HD 107146` [Mesa, et al. 2021MNRAS.503.1276M](https://ui.adsabs.harvard.edu/abs/2021MNRAS.503.1276M/abstract)

Credits
-------
The Exo-DMC is the latest (although the first one in Python) rendition of the `MESS` (Multi-purpose Exoplanet Simulation System).

To understand the DMC's underlying assumptions is therefore useful to read about the `MESS` in its various iteration: 

* MESS (Multi-purpose Exoplanet Simulation System) [Bonavita et al.  2012, A&A, 537, A67](https://arxiv.org/abs/1110.4917): first version of the code (note that the link provided in the paper is not working anymore)
* Quick-MESS: A Fast Statistical Tool for Exoplanet Imaging Surveys [Bonavita et al.  2013, PASP, 125, 849](https://arxiv.org/abs/1306.0935): quick version of MESS, which abandones the Monte Carlo approach for a faster grid-like one. 
* MESS2: [Lannier et al.  2017 A&A, 603, A54](https://arxiv.org/abs/1704.07432): designed to combined multiple data sets, both from Direct Imaging and Radial Velocity. 

Like MESS, the DMC allows for a high level of flexibility in terms of possible assumptions on the synthetic planet population to be used for the determination of the detection probability. 

Although the present version is a very basic one, you can have a glimpse of what's to come by checking out some of the analysis performed with `MESS`, `QMESS` and `MESS2`:

* Constraints on gian planet occurrence rate 
  * for single stars [Stone et al. 2018, AJ, 156, 286](https://arxiv.org/abs/1810.10560)
  * or binary stars [Bonavita et al. 2016, A&A, 593, A38](https://arxiv.org/abs/1605.03962), (Bonavita & Desidera 2020, Galaxies 2020, 8, 16)[https://arxiv.org/abs/2002.11734]
* Constraints on planet formation models 
    * [Humphries et al. 2019, MNRAS, 488, 4873](https://arxiv.org/abs/1907.07584)
    * [Vigan et al. 2017, A&A, 603, A3](https://arxiv.org/abs/1703.05322)
* Constraints on brown dwarf variability [Vos et al. 2019, MNRAS, 483, 480](https://arxiv.org/abs/1004.3487)
* Constraints on specific objects: [Bonnefoy et al. 2021, A&A, 655A, 62](https://ui.adsabs.harvard.edu/abs/2021A%26A...655A..62B/abstract), [Bonavita et al. 2010, A&A, 522, A2](https://arxiv.org/abs/1004.3487)



Author and contributors
-----------------------

Mariangela Bonavita <[mariangela.bonavita@open.ac.uk](mailto:mariangela.bonavita@open.ac.uk)>, The Open University, UK 

With important contributions from:
* Silvano Desidera (INAF-OAPD)
* Ernst de Moij (CfAR)
* Arthur Vigan (LAM / CNRS)
* Justine Lannier 

We are grateful for your effort, and hope that these tools will contribute to your scientific work and discoveries. Please feel free to report any bug or possible improvement to the author(s).

Attribution
-----------------------
Please cite [Bonavita 2020](https://ui.adsabs.harvard.edu/abs/2020ascl.soft10008B/abstract) whenever you publish results obtained with the Exo-DMC. Astrophysics Source Code Library reference [ascl:2010.008](https://ascl.net/2010.008)
