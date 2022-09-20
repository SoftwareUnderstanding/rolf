# runDM

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.823249.svg)](https://doi.org/10.5281/zenodo.823249) [![arXiv](https://img.shields.io/badge/arXiv-1605.04917-B31B1B.svg)](http://arxiv.org/abs/1605.04917) [![MIT Licence](https://badges.frapsoft.com/os/mit/mit.svg?v=103)](https://opensource.org/licenses/mit-license.php)

*With runDMC, It's Tricky. With runDM, it's not.*

`runDM` is a tool for calculating the running of the couplings of Dark Matter (DM) to the Standard Model (SM) in simplified models with vector mediators. By specifying the mass of the mediator and the couplings of the mediator to SM fields at high energy, the code can be used to calculate the couplings at low energy, taking into account the mixing of all dimension-6 operators. The code can also be used to extract the operator coefficients relevant for direct detection, namely low energy couplings to up, down and strange quarks and to protons and neutrons. Further details about the physics behind the code can be found in Appendix B of [arXiv:1605.04917](http://arxiv.org/abs/1605.04917).

At present, the code is written in two languages: *Mathematica* and *Python*. If you are interested in an implementation in another language, please get in touch and we'll do what we can to add it. But if you want it in Fortran, you better be ready to offer something in return. Installation instructions and documentation for the code can be found in `doc/runDM-manual.pdf`. We also provide a number of example files:

- For the Python code, we provide an example script as well as Jupyter Notebook. A static version of the notebook can be viewed [here](http://nbviewer.jupyter.org/github/bradkav/runDM/blob/master/python/runDM-examples.ipynb).

- For the Mathematica code, we provide an example notebook. We also provide an example of how to interface with the [NRopsDD code](http://www.marcocirelli.net/NROpsDD.html) for obtaining limits on general models.

If you make use of `runDM` in your own work, please cite it as:

>F. D’Eramo, B. J. Kavanagh & P. Panci (2016). runDM (Version X.X) [Computer software], doi:10.5281/zenodo.823249. Available at https://github.com/bradkav/runDM/

making sure to include the correct version number. Please also cite the associated papers:

>A. Crivellin, F. D’Eramo & M. Procura, New Constraints on Dark Matter Effective Theories from Standard Model Loops, Phys. Rev. Lett. 112 (2014) 191304 [arXiv:1402.1173],

>F. D’Eramo & M. Procura, Connecting Dark Matter UV Complete Models to Direct Detection Rates via Effective Field Theory, JHEP 1504 (2015) 054 [arXiv:1411.3342],

>F. D’Eramo, B. J. Kavanagh & P. Panci, You can hide but you have to run: direct detection with vector mediators, JHEP 1608 (2016) 111 [arXiv:1605.04917].

Please contact Bradley Kavanagh (bradkav@gmail.com) for any questions, problems, bugs and suggestions.

------------

`runDM` has been used in the following publications:

- B. Barman et al., Catch 'em all: Effective Leptophilic WIMPs at the e+e− Collider, [arXiv:2109.10936](https://arxiv.org/abs/2109.10936)  
- S. Basegmez du Pree, Robust Limits from Upcoming Neutrino Telescopes and Implications on Minimal Dark Matter Models, [arXiv:2103.01237](https://arxiv.org/abs/2103.01237)  
- I. Bischer, T. Plehn, W. Rodejohann, Dark Matter EFT, the Third -- Neutrino WIMPs, [arXiv:2008.04718](https://arxiv.org/abs/2008.04718)
- Q.-H. Cao, A.-K. Wei, Q.-F. Xiang, Dark Matter Search at Colliders and Neutrino Floor, [arXiv:2006.12768](https://arxiv.org/abs/2006.12768)  
- W. Chao, Direct detections of Majorana dark matter in vector portal, [arXiv:1904.09785](https://arxiv.org/abs/1904.09785)  
- M. Arteaga et al., Flavored Dark Sectors running to low energy, [arXiv:1810.04747](https://arxiv.org/abs/1810.04747)  
- S. Kang et al., On the sensitivity of present direct detection experiments to WIMP-quark and WIMP-gluon effective interactions: a systematic assessment and new model-independent approaches, [arXiv:1810.00607](https://arxiv.org/abs/1810.00607)  
- B. J. Kavanagh, P. Panci, R. Ziegler, Faint Light from Dark Matter: Classifying and Constraining Dark Matter-Photon Effective Operators, [arXiv:1810.00033](https://arxiv.org/abs/1810.00033)  
- A. Belyaev et al., Interplay of the LHC and non-LHC Dark Matter searches in the Effective Field Theory approach, [arXiv:1807.03817](https://arxiv.org/abs/1807.03817)
- A. Falkowski et al., Flavourful Z' portal for vector-like neutrino Dark Matter and RK(*), [arXiv:1803.04430](https://arxiv.org/abs/1803.04430)
- G. Bertone et al., Identifying WIMP dark matter from particle and astroparticle data, [arXiv:1712.04793](https://arxiv.org/abs/1712.04793)
- G. H. Duan et al., Simplified TeV leptophilic dark matter in light of DAMPE data, [arXiv:1711.11012](https://arxiv.org/abs/1711.11012)
- S. Baum et al., Determining Dark Matter properties with a XENONnT/LZ signal and LHC-Run3 mono-jet searches, [arXiv:1709.06051](https://arxiv.org/abs/1709.06051)
- E. Bertuzzo, C. J. Caniu Barros, G. Grilli di Cortona, MeV Dark Matter: Model Independent Bounds, [arXiv:1707.00725](https://arxiv.org/abs/1707.00725)
- Y. Cui, F. D'Eramo, Surprises from Complete Vector Portal Theories: New Insights into the Dark Sector and its Interplay with Higgs Physics, [arXiv:1705.03897](https://arxiv.org/abs/1705.03897)
- L. Roszkowski, S. Trojanowski, K. Turzynski, Towards understanding thermal history of the Universe through direct and indirect detection of dark matter, [arXiv:1703.00841](https://arxiv.org/abs/1703.00841)
- F. D'Eramo, B. J. Kavanagh, P. Panci, Probing Leptophilic Dark Sectors with Hadronic Processes, [arXiv:1702.00016](https://arxiv.org/abs/1702.00016)
- A. Celis, W.-Z. Feng, M. Vollmann, Dirac dark matter and b→sℓ+ℓ− with U(1) gauge symmetry, [arXiv:1608.03894](https://arxiv.org/abs/1608.03894)
- F. D'Eramo, B. J. Kavanagh, P. Panci, You can hide but you have to run: direct detection with vector mediators, [arXiv:1605.04917](http://arxiv.org/abs/1605.04917)
