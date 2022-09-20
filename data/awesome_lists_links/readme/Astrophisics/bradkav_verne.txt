# Verne

[![DOI](https://zenodo.org/badge/112917758.svg)](https://zenodo.org/badge/latestdoi/112917758) [![arXiv](https://img.shields.io/badge/arXiv-1712.04901-B31B1B.svg)](https://arxiv.org/abs/1712.04901) [![MIT Licence](https://badges.frapsoft.com/os/mit/mit.svg?v=103)](https://opensource.org/licenses/mit-license.php)


Verne is a python code for calculating the Earth-stopping effect for Dark Matter (DM). 

The code allows you to calculate the speed distribution (and DM signal rate) at detectors in the Earth's atmosphere, on the Earth's surface and at underground detector locations, such as MPI (the Max-Planck-Institute in Munich) and SUF (the Stanford Underground Facility). This can be done for a range of DM masses and cross sections (though the results will be most reliable for very heavy DM particles). This is because the code implements a 'continuous stopping' formalism, with DM particles travelling on straight-line trajectories. Further details about the physics behind the code can be found in [arXiv:1712.04901](https://arxiv.org/abs/1712.04901).

The core of the code (the `verne` module) is in the [src](src) folder, head there for some details of how to use the code.


### Version history

**Version 1.3 (30/08/2022):** Major update to include millicharged DM interactions.  
**Version 1.2 (09/03/2021):** Major update to include spin-dependent interactions (and fix a few minor bugs). Now compatible with Python3.  
**Version 1.1 (12/02/2018):** Updated event rate calculation to account for correct CRESST exposure times. Minor edits to text.  
**Version 1.0 (14/12/2017):** Initial release (including arXiv numbers, etc.)  
**Version 0.9 (13/12/2017):** Pre-release before arXiv submission.  

### Contents

- `src`: Core of the `verne` code, including calculation scripts.
- `data`: A collection of data tables (Earth element density profiles, etc.) which are read in by `verne`. 
- `results`: Numerical results for the final speed distributions, numbers of signal events and constraints.
- `plotting`: Scripts for generating plots from the results.
- `plots`: Plots and illustrations summarising the key results. Mostly from [arXiv:1712.04901](https://arxiv.org/abs/1712.04901).  
- `paper`: PDF and tex files for the associated paper.

### Getting started

The best place to start is probably the [example notebook](/src/Example.ipynb). You can also find more detailed info in the README file for the [`src/`](src/) folder. I'm also very happy to provide more specific examples if you have something in particular in mind - just get in touch. 

### Dependencies

The code is compatible with Python3. Requires [numpy](http://www.numpy.org) and [scipy](https://www.scipy.org). More detailed dependencies can be found in `requirements.txt`.

Some code is included for calculating nuclear recoil spectra in direct detection experiments, but we recommend using [`WIMpy`](https://github.com/bradkav/WIMpy_NREFT) (which is more flexible and more complete) for recoil spectra calculations.


### Citing Verne

If you make use of the code or the numerical results, please cite:

>Kavanagh, B. J., 2016, Verne, Astrophysics Source Code Library, record ascl:1802.005, available at https://github.com/bradkav/verne.

as well as citing the original paper, [arXiv:1712.04901](https://arxiv.org/abs/1712.04901):

>Kavanagh, B. J. (2017), "Earth-Scattering of super-heavy Dark Matter: updated constraints from detectors old and new", arXiv:1712.04901.

### Papers

The `verne` code has been used (in some form or another) in the follow research papers:
- *Search for sub-GeV Dark Matter via Migdal effect with an EDELWEISS germanium detector with NbSi TES sensors*, EDELWEISS Collaboration & B. J. Kavanagh (2022), [arXiv:2203.03993](https://arxiv.org/abs/2203.03993)
- *Etching Plastic Searches for Dark Matter*, A. Bhoonah, J. Bramante, B. Courtman & N. Song (2020), [arXiv:2012.13406](https://arxiv.org/abs/2012.13406)  
- *Detecting Composite Dark Matter with Long Range and Contact Interactions in Gas Clouds*, A. Bhoonah, J. Bramante, S. Schon & N. Song (2020), [arXiv:2010.07240](https://arxiv.org/abs/2010.07240)  
- *Light Dark Matter Search with a High-Resolution Athermal Phonon Detector Operated Above Ground*, Alkhatib et al (SuperCDMS Collaboration, 2020), [arXiv:2007.14289](https://arxiv.org/abs/2007.14289)
- *Migdal effect and photon Bremsstrahlung: improving the sensitivity to light dark matter of liquid argon experiments*, G. Grilli di Cortona, A. Messina & S. Piacentini (2020), [arXiv:2006.02453](https://arxiv.org/abs/2006.02453)  
- *Measuring the local Dark Matter density in the laboratory*, B. J. Kavanagh, T. Emken & R. Catena (2020), [arXiv:2004.01621](https://arxiv.org/abs/2004.01621)  
- *A Search for Light Dark Matter Interactions Enhanced by the Migdal effect or Bremsstrahlung in XENON1T*, Xenon1T Collaboration (2019), [arXiv:1907.12771](https://arxiv.org/abs/1907.12771)
- *Searching for low-mass dark matter particles with a massive Ge bolometer operated above-ground*, EDELWEISS Collaboration & B. J. Kavanagh (2019), [arXiv:1901.03588](https://arxiv.org/abs/1901.03588)  
- *Earth-Scattering of super-heavy Dark Matter: updated constraints from detectors old and new*, B. J. Kavanagh (2017), [arXiv:1712.04901](https://arxiv.org/abs/1712.04901)  


