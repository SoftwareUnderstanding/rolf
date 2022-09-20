# 21CMMC
A parallelised, Monte Carlo Markov Chain analysis tool for simultaneously exploring 
the epoch of reionisation (EoR) and the epoch of heating

Note: The latest release is an intermediate release to coincide with the latest parametrisation (http://adsabs.harvard.edu/cgi-bin/bib_query?arXiv:1809.08995).
A newer, cleaner and more intuitive version is currently under development and will be available soon.

Intermediate release: 25/09/2018
- New astrophysical parameterisation for the ionising sources mimicking more closely galaxy properties
- By construction this provides UV ionising luminosity functions capable of matching observational data
- Allows joint sampling of the 21cm signal and UV luminosity functions to improve constraining power

Release: 08/01/2018
- Updated to include spin temperature fluctuations and the heating epoch
- Added in line-of-sight redshift space distortions
- Added in support for light-cone generation
- Allow both astrophysical and cosmological parameters to vary

21CMMC is a fast and efficient Python sampler of the semi-numerical reionisation simulation code 21cmFAST. It 
can recover constraints on astrophysical parameters from current or future 21 cm EoR experiments, accommodating 
a variety of EoR models, as well as priors on individual model parameters and the reionisation history. 
By studying the resulting impact on the EoR astrophysical constraints, 21CMMC can be used to optimise: (i) 
interferometer designs; (ii) foreground cleaning algorithms; (iii) observing strategies; (iv) alternate statistics 
characterising the 21cm signal; and (v) synergies with other observational programs. 

For more details please see:
(1) http://adsabs.harvard.edu/abs/2015MNRAS.449.4246G (original 21CMMC)
(2) http://adsabs.harvard.edu/abs/2017MNRAS.472.2651G (adding in spin temperature fluctuations and the heating epoch)
(3) http://adsabs.harvard.edu/abs/2018arXiv180101592G (added light-cones and line-of-sight redshift space distortions)
(4) http://adsabs.harvard.edu/cgi-bin/bib_query?arXiv:1809.08995 (added new astrophysical parameterisation and ability to compute UV luminosity functions)

For a more detailed readme, see the files in the Documentation folder within the 21CMMC_SourceCode directory.

Any questions, comments or feedback (including technical support and setting up) should be directed 
to me (Bradley Greig) at brad.s.greig@gmail.com. If you use this code in any of your work, please cite the above papers 
and link to this repository. 
