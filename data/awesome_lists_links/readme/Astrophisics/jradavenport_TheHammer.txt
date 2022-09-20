The Hammer: An IDL Spectral Typing Suite
=========

**The Hammer** spectral typing algorithm was originally developed for use on late-type SDSS spectra, but has subsequently been modified to allow it to classify spectra in a variety of formats with targets spanning the MK spectral sequence. The Hammer processes a list of input spectra by automatically estimating each object's spectral type and measuring activity and metallicity tracers in late type stars. Once automatic processing is complete, an interactive interface allows the user to manually tweak the final assigned spectral type through visual comparison with a set of templates.


The distribution provided above contains a [README](README) file to explain the usage of routine, as well as a small set of spectra and associated output files to demonstrate the usage of the code. Also available are tables outlining the indices used by the hammer: single numerator indices are presented in [table1.txt](resources/table1.txt), multiple numerator indices are presented in [table2.txt](resources/table2.txt). For the definition of how each type of index is measured, as well as a full overview of the Hammer algorithm, see Appendix A of [Covey et al. 2007](http://adsabs.harvard.edu/abs/2007AJ....134.2398C).

**Screenshot of the Hammer's interactive spectral typing mode:**
</br> ![](HammerScreenShot.png?raw=true =350x)
