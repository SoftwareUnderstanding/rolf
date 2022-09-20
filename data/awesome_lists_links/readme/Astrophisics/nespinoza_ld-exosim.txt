# ld-exosim

This repository stores codes to (1) select the optimal (i.e. best estimator in a MSE 
sense) limb-darkening law for a given transiting exoplanet lightcurve and (2) calculate 
the limb-darkening induced biases on various exoplanet parameters. 

The details of the codes are explained in Espinoza & Jordán (2016, MNRAS, in press.; 
arXiv e-print: http://arxiv.org/abs/1601.05485). Source code of the paper (including 
generation of all figures): https://github.com/nespinoza/lds_which_law_2_use. 

DEPENDENCIES
------------

This code makes use of three important libraries:

    + The Bad-Ass Transit Model cAlculatioN (batman) package: http://astro.uchicago.edu/~kreidberg/batman/
    + The latest version of the lmfit fitter (https://lmfit.github.io/lmfit-py/)
    + The LDC3.py code wrote by David Kipping (https://github.com/davidkipping/LDC3)

This last code might be updated with time, but I have copied here the October 29th, 2015 version of it
for reference: be sure to use the latest version of D. Kipping's code!

USAGE
------------
The usage of the code is simple, depending on what you want to do:

1. **Do you want to know which law to use in a given application?**

   You are looking for the `which_law_should_i_use.py` code. Simply modify 
   the parameters inside the code and let the simulations run. At the end, 
   the code will print out the Bias/Precision values for each law so you can 
   select the optimal one for your application.

2. **You want to perform bias simulations for several transit parameters?**

   Then you want to use the `run_ld_exosim.py` code. In the code just define 
   the parameters you woud like to explore and run it. The results will be 
   saved in a folder named "results" for your simulation, where the biases 
   for both fixed and free parameters will be stored. 

Both codes make use of limb-darkening tables stored in the `ld_tables` folder, 
which already has a table containing all the limb-darkening coefficients using 
the ATLAS models and the Kepler bandpass. To generate your own table, you can use 
our code at https://github.com/nespinoza/limb-darkening and put the result inside.

