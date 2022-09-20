# satcand
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4026288.svg)](https://doi.org/10.5281/zenodo.4026288)

Tools to apply theoretical constraints of orbital stability and tidal migration to KOI exomoon candidates.  

This is a repository for tools to apply known theoretical constraints of orbital stability and tidal migration to KOI exomoon candidates, which reproduce the figures in Quarles, Li, & Rosario-Franco (2020). Orbital stability analysis from [Rosario-Franco et al. (2020)](https://ui.adsabs.harvard.edu/abs/2020AJ....159..260R/abstract) can provide the critical semimajor axis (a_c) of exomoons in terms of the host planet's Hill radius R_H (in AU), where this can be scaled/converted into units of the planetary radius R_p (in AU). Using our knowledge of the solar system planets, we can evaluate the orbital evolution due to tidal migration considering a constant Q tidal model (e.g., [Sasaki, Barnes & O'Brien (2012)](https://ui.adsabs.harvard.edu/abs/2012ApJ...754...51S/abstract)). In addition to the theoretical constraints, observational constraints can be applied using results from [Kipping (2020)](https://ui.adsabs.harvard.edu/abs/2020arXiv200803613K/abstract).

Included in this repository are: 
* KOI_stab.py
  * produces Figure 1 from Quarles, Li, & Rosario-Franco (2020)
* tidal_migration.py
  * evaluates the tidal migration within a Sun-Earth-Moon system using initial conditions from [Sasaki, Barnes & O'Brien (2012)](https://ui.adsabs.harvard.edu/abs/2012ApJ...754...51S/abstract) (see their Fig. 1)
  * plots angular velocity over time, which includes the time evolution of the spin angular momentum of the planet (omega_p), the orbital mean motion of the planet n_p, and the orbital mean motion of the satellite n_sat (SBO_tide_evol.png).  Both T1 and T are calculated and plotted as vertical within the angular velocity evolution plot.
* SBO_tidal_tree.py
  * implements the decision tree algorithm from [Sasaki, Barnes & O'Brien (2012)](https://ui.adsabs.harvard.edu/abs/2012ApJ...754...51S/abstract) to calculate the migration time scale (T1) and the total migration time scale (T). 
* Calc_tidal_limit.py
  * demonstrates the calculation of the minimum Q_p so that stable moon parameters (mass and separation) can be inferred
* plot_Q_crit.py
  * produces Figure 2 from Quarles, Li, & Rosario-Franco (2020) using output from Calc_tidal_limit.py
* plot_tide_evol.py
  * produces Figure 3 from Quarles, Li, & Rosario-Franco (2020) using the output in KOI1925 folder
  * assumes that the satellite is initially separaterd by 5 R_p and the host planet spin period is 10 hours
* plot_combined_constraints.py
  * combines theoretical and observational constraints for the 6 KOI candidates proposed by [Fox & Wiegert (2020)](https://ui.adsabs.harvard.edu/abs/2020arXiv200612997F/abstract).
  * uses 3\sigma curves from [Kipping (2020)](https://ui.adsabs.harvard.edu/abs/2020arXiv200803613K/abstract)
  * uses output from Calc_tidal_limit.py
  * produces Figure 4 from Quarles, Li, & Rosario-Franco (2020)

These python scripts assume that the basic dependencies (e.g., Numpy, Scipy, Matplotlib) are already installed or the user is in an Anaconda environment.

# Attribution
---------------
Please use the following citation, if you find these tools useful in your research.
```
@ARTICLE{Quarles2020,
       author = {{Quarles}, Billy and {Li}, Gongjie and {Rosario-Franco}, Marialis},
        title = "{Application of Orbital Stability and Tidal Migration Constraints for Exomoon Candidates}",
      journal = {\apjl},
     keywords = {Exoplanet dynamics, Exoplanet tides, Natural satellites (Extrasolar), Exoplanet systems, 490, 497, 483, 484, Astrophysics - Earth and Planetary Astrophysics},
         year = 2020,
        month = oct,
       volume = {902},
       number = {1},
          eid = {L20},
        pages = {L20},
          doi = {10.3847/2041-8213/abba36},
archivePrefix = {arXiv},
       eprint = {2009.14723},
 primaryClass = {astro-ph.EP},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2020ApJ...902L..20Q},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
