# SNSED_Repository
Repository of extrapolated Core-Collapse supernova (CC SN) SEDs and an extended version of the Type Ia SALT2 model, constructed with the [SNSEDextend](https://github.com/jpierel14/snsed) Python package (for help, see the [SNSED ReadTheDocs](http://snsedextend.readthedocs.io/en/latest/) pages).

## SEDs.P18-UV2IR (snsedextend_v1.0)
_**Currently the most up-to-date version of the extrapolated SED library.**_

Sub-directories `Type_Ib`, `Type_Ic`, and `Type_II` contain .SED files for all the extrapolated CC SN SED templates.  The `Type_Ia` sub-directory contains the extrapolated SALT2 model, extrapolated to UV and NIR wavelengths, but not re-trained for light curve fitting to derive cosmological distance measurements.  This version of the SED extrapolation is presented in Pierel et al. 2018 (in prep).

When included in the [SNANA](http://snana.uchicago.edu/) SNDATA_ROOT installation, these models will be labeled as NON1A.P18-UV2IR and SALT2.P18-UV2IR. 

## SEDs.H18-WFIRST (snsedextend_v0)
The version of extrapolated CC SN SEDs and the Type Ia SALT2 model that were used for simulations of the WFIRST SN survey in [Hounsell et al. 2017](https://ui.adsabs.harvard.edu/#abs/2017arXiv170201747H/abstract) (accepted for publication in ApJ). Sub-directory `NON1A.H18-WFIRST` contains the non-Ia (i.e., CC SN) SED templates, and `SALT2.H18-WFIRST` contains the modified SALT2 model.  See section 4 of the Hounsell et al. 2017 paper for an explanation of the extrapolation process.   The python modules used for the extrapolations (`snsedextend.py` and `salt2ir.py`) are included in the sub-directories. 

Note that an earlier but very similar version of the template CCSN SEDs and SALT2 model stored here were also used for analysis of the CANDELS and CLASH high-redshift SN survey results, presented in [Rodney et al. 2014](https://ui.adsabs.harvard.edu/#abs/2014AJ....148...13R/abstract), [Graur et al. 2014](https://ui.adsabs.harvard.edu/#abs/2014ApJ...783...28G/abstract), and [Strolger et al. 2015](https://ui.adsabs.harvard.edu/#abs/2015ApJ...813...93S/abstract).

In the [SNANA](http://snana.uchicago.edu/) SNDATA_ROOT installation, these models are labeled as NON1A.H17-WFIRST and SALT2.H17-WFIRST. 

## Figures
Sub-directories for each CC SN sub-type have .pdf files showing the extrapolated SED, at peak brightness.  The Type_Ia sub-directory contains .pdf images showing the extrapolation of all the SALT2 model components.  Examples linked below. 

![Extrapolated SED of CSP-2006ep (Type Ib)](Figures/TypeIb/CSP-2006ep.pdf)

![SALT2 M0 extrapolation to IR at peak](Figures/TypeIa/SALT2ir_template0_peak.pdf)
