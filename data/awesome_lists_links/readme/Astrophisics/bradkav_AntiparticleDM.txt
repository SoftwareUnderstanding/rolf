# AntiparticleDM

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.815457.svg)](https://doi.org/10.5281/zenodo.815457) [![arXiv](https://img.shields.io/badge/arXiv-1706.07819-B31B1B.svg)](https://arxiv.org/abs/1706.07819) [![MIT Licence](https://badges.frapsoft.com/os/mit/mit.svg?v=103)](https://opensource.org/licenses/mit-license.php)

Python code for calculating the prospects of future direct detection experiments to discriminate between Majorana and Dirac Dark Matter (i.e. to determine whether Dark Matter is its own antiparticle). Direct detection event rates and mock data generation are taken care of by a variation of the `WIMpy` code (also available [here](https://github.com/bradkav/WIMpy/tree/Antiparticle)).

With this code, the results of [arXiv:1706.07819](https://arxiv.org/abs/1706.07819) should be ***entirely reproducible***. Follow the instructions [below](#repro) if you want to reproduce those results. If you find any mistakes or have any trouble at all reproducing any of the results, please open an issue or get in touch directly.

If you have any questions, comments, bug-reports etc., please contact Bradley Kavanagh at bradkav@gmail.com. 

#### Version History

**Version 1.0.3 (15/09/2017):** Added script for plotting illustration of fundamental couplings. Code should now match arXiv-v2.  
**Version 1.0.2 (06/07/2017):** Updated results after fixing some minor bugs.  
**Version 1.0.1 (27/06/2017):** Added arXiv number and fixed a couple of typos.  
**Version 1.0.0 (23/06/2017):** Initial release, including all results and plots from the paper.

## Contents

- `calc`: core code for calculating the statistical significance for discriminating between Dirac and Majorana Dark Matter (DM).
- `scripts`: scripts for reproducing results from the paper (NB: some may need to be implemented on a computing cluster...)
- `analysis`: scripts for processing the results and generating plots.
- `results`: data products for a range of DM masses, couplings and experimental ensembles.
- `plots`: plots from [arXiv:1706.07819](https://arxiv.org/abs/1706.07819) (and others).

## Reproducing the results <a name="repro"></a>

The majority of the code is written in `python`, and requires the standard `numpy` and `scipy` libraries. For plotting, `matplotlib` is also required. Code for generating mock data sets and performing likelihood fits are found in the `calc` folder. Check the README in the `calc` folder for (slightly) more detail on how it works.

#### Performing likelihood fits

For calculating the discrimination significance for a single point in parameter space, check out the jupyter notebook [`calc/index.ipynb`](calc/index.ipynb).

For calculating the discrimination significance over a grid of the input couplings, run the example script  [`scripts/RunFits_couplings.sh`](scripts/RunFits_couplings.sh). 

For calculating the discrimination significance as a function of exposure (for a fixed input), run the example [`scripts/RunFits_exposure.sh`](scripts/RunFits_exposure.sh).

Note that these scripts will take a long time to run (think hours to days...). In practice then, you'll probably want to run things on a computing cluster. For this, we provide two python files [`scripts/RunMPI_couplings.py`](scripts/RunMPI_couplings.py) and [`scripts/RunMPI_exposure.py`](scripts/RunMPI_exposure.py), which are MPI-enabled and take care of running large numbers of fits in parallel. To use these, `mpi4py` is required.

#### Generating plots

Scripts for generating plots from the results are in the `analysis/` folder. To (re-)generate all the plots from the paper, simply run `scripts/GeneratePlots.sh`.

#### Checking the likelihood calculator

You can also check that the likelihood calculator works well by running 

```python
python CompareNgrid.py mx
```

in the `calc` folder (where `mx` is the DM mass in GeV). This will calculate the maximum likelihood as a function of mass for different densities of grid (showing hopefully that the case of a 50x50x50 grid for Dirac DM works well). There are also some plots to this effect in the `plots` folder.

We also ran some calculations of the discrimination significance using realistic isotope distributions, to compare with the simple 'single-isotope' approximations used in the paper. In the `results` folder, these are listed with `full` after the ensemble. The plot `plots/Exposure_R=0.75_comparison.pdf` shows the results.

## Citation

If you make use of the code or the numerical results, please cite the project as:

> Kavanagh, B. J., Queiroz, F. S., Rodejohann, W., Yaguna, C. E., "AntiparticleDM" (2017), https://github.com/bradkav/AntiparticleDM/, [doi:10.5281/zenodo.815457](http://dx.doi.org/10.5281/zenodo.815457)

Please also cite the associated papers:

> Kavanagh, B. J., Queiroz, F. S., Rodejohann, W., Yaguna, C. E., "Prospects for determining the particle/antiparticle nature of WIMP dark matter with direct detection experiments" (2017), [arXiv:1706.07819](https://arxiv.org/abs/1706.07819)

> Queiroz, F. S., Rodejohann, W., Yaguna, C. E., "Is the dark matter particle its own antiparticle?", Phys. Rev. D 95 (2017) 095010, [arXiv:1610.06581](https://arxiv.org/abs/arXiv:1610.06581)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
