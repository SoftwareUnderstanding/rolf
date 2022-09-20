# simuTrans
Exoplanet transit simulator.

INSTALL: 

dependence: libgsl 

clone the noprecession branch

make 

Example 

To check the inital condition: 

python fit_lc.py -c tres-2b.cfg --plot

python fit_lc.py -c koi368_bestfit.cfg --plot

python fit_lc.py -c koi368_gd_bestfit.cfg --plot

To run MCMC

python fit_lc.py -c tres-2b.cfg
