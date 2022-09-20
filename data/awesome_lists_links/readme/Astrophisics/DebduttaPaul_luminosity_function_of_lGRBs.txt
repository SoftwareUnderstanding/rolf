# luminosity_function_of_lGRBs
This is the set of data files and codes used for the project Paul, D. 2018, MNRAS, 473, 3385 (https://ui.adsabs.harvard.edu/#abs/2018MNRAS.473.3385P). Please cite this paper if you are using any of these codes or databases.

The following is a brief description of the codes, as well as the databases used and created.


debduttaS_functions.py        : A set of generic functions.
specific_functions.py         : A set of functions used for this work.

k_correction.py               : Calculates the k-factor for both Swift and Fermi, for GRBs with known/fixed spectral parameters.
k_table.txt                   : Tabulates the k-factor for fixed spectral parameters as a function of redshift.

Fermi--all_GRBs.txt           : The Fermi data from the Fermi catalogue.
Swift--all_GRBs.txt           : The Swift data from the Swift catalogue.
Swift--GRBs_with_redshifts    : The Swift data for GRBs with known redshifts, from the Swift catalogue.

rho_star_dot.txt              : The numerical values of the Cosmic Star formation rate, from Bouwens et al. (2015).

selecting_common_GRBs--all.py : Selects all the GRBs common to both Swift and Fermi.
selecting_common_GRBs--wkr.py : SElects GRBs with measured redshift (by Swift), common to both Swift and Fermi.
common_GRBs--all.txt          : Output of "selecting_common_GRBs--all.py".
common_GRBs--wkr.txt          : Output of "selecting_common_GRBs--wkr.py".

establishing_the_correlation--step1--Tan_method--blind.py             : Attempt parameter estimation as in Tan et al. (2013).
establishing_the_correlation--step2--Tan_method--corrected.py         : Correct the modification procedure and check.
establishing_the_correlation--step3--looking_for_possible_systematics : Try to explain the discrepancy.

estimating_pseudo_redshifts_and_Luminosities--with_names.py : Segregate the classes of GRBs and estimate pseudo redshifts.
L_vs_Z--Fermi_long---with_names.txt                         : Output of above code for the "Fermi GRBs" (Table 1), long.
L_vs_Z--Fermi_short--with_names.txt                         : Output of above code for the "Fermi GRBs" (Table 1), short.
L_vs_Z--Swift_long---with_names.txt                         : Output of above code for the "Swift GRBs" (Table 1), long.
L_vs_Z--Swift_short--with_names.txt                         : Output of above code for the "Swift GRBs" (Table 1), short.

sensitivity_plots.py          : Plotting the above data along with the computed instrumental thresholds.
thresholds.txt                : One-time output of above code, used for all consecutive runs of the code.

fitting_the_phi--ECPL.py          : Exploring the parameter space of the ECPL model.
fitting_the_phi--ECPL--plots.py     : Plotting the fits for the solutions of the ECPL model.
fitting_the_phi--BPL.py             : Exploring the parameter space of the BPL model.
fitting_the_phi--BPL--plots.py      : Plotting the fits for the solutions of the BPL model.
ratio_of_models.py                  : Explaining discrepancy of data and model at high redshfts from a simple hypothesis.
parameter_error_estimation--BPL.py  : Estimating the errors in the final solutions of the BPL model.
parameter_error_estimation--ECPL.py : Estimating the errors in the final solutions of the ECPL model.

