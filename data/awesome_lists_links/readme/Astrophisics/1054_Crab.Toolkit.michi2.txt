# Crab.Toolkit.michi2
This is a _Crab_ toolkit for minimumizing chi2 for any 1D model fit, for example, galaxy SED fitting, molecular line ladder fitting. 

# Usage for SED Fitting #

## SED Fitting ##

Our SED fitting contains 5 components (or any number of components): BC03, AGN, DL07 warm and cold and radio. The radio are constrainted by the DL07 warm+cold summed rest-frame 8-1000um luminosity, so it is not independent. If you do not want that, you can simply do not include radio data in the SED fitting input file. 

The fitting needs a specific redshift as the input, so it can not directly fit redshift... 

The advantage of this code is that it loops over all DL07 dust models and combinations with AGN models and stellar models, so can fit ISRF and Mdust in a finer parameter grid. 


### 1. Download the code
Scroll up, there is a green button **"Clone or download"**. Click it and select **"Download ZIP"**. Download it to somewhere, for example, your working directory, and uncompress it. 

Currently the ZIP file size is about 52MB. 

Or if you are familiar with git, you can clone it, but note that the repository size is a bit large, like hundreds of MB.  

### 2. Source the code (must under BASH shell)
```
bash
source /some/path/Crab.Toolkit.michi2/SETUP.bash
```

### 3. Prepare photometry data
Assuming we have a photometric catalog, where columns are different photometric bands and rows are different sources. Then we need to prepare one SED fitting input file for each source you would like to fit. Each SED fitting input file should be a text file and has three columns: first column the wavelength in micron-meter unit, second column the flux density in milli-Jansky unit, and the third column the error in flux density in milli-Jansky unit. It should have multiple rows corresponding to each photometric band, and better S/N>3 bands. Columns should be separated by white space. Rows can be commented by # character. 

An example of the SED fitting input file is like: 
```
# wavelength_um     flux_mJy    flux_err_mJy
    3.56343       0.12417548     0.012417548
    4.51101       0.09488227     0.009488227
    5.75934      0.075890755    0.0075890755
    7.95949       0.08685837     0.009947749
       24.0        1.3679016      0.13679016
      100.0        41.577702       4.1577702
      160.0          60.0042       6.3393602
      250.0          58.0821         5.80821
      350.0        36.100101       6.0408401
      500.0        18.878901       2.2385001
      850.0         3.866907        1.258796
```

### 4. Run michi2
If you have already `sourced` the `SETUP.bash`, then just change directory to where you store your SED fitting input file (assuming it's named "extracted_flux.txt"), and run michi2.

```
cd /path/to/your/data/directory/

ls "extracted_flux.txt"

michi2-run-SED-fitting-v5 # call it without any argument will print the usage

michi2-run-SED-fitting-v5 -redshift 1.5 -flux "extracted_flux.txt" -parallel 2

# Note that for this example we set redshift to 1.5, and fit with 2 CPU cores.

# Some optional arguments below

michi2-run-SED-fitting-v5 -redshift 1.5 -flux "extracted_flux.txt" -parallel 2 -sampling 150000 -lib-stellar BC03.MultiAge -lib-dust DL07UPD2010 -lib-AGN MullaneyAGN -lib-radio Radio -freeze-radio -qIR 2.4 -Umin 1 -minEBV 0.2 -obj-name "My Galaxy" -overwrite
```

[comment]: <> (The michi2 SED fitting is currently **VERY SLOW**. It can easily take three hours on a laptop! It is because it stupidly loops over all the combinations of all input models, so if you fit with 5 components, it takes hours and hours. Currently we parallized it. We will adopt Markov chain Monte Carlo method in the future. For now, for our own, we use it on 100plus-CPU-core machine, so it is still fine.) 

The output of michi2 SED fitting will be: 
```
fit_5.out        # a text file, containing chi-square and parameters of each combination of components
fit_5.out.info   # a text file, containing basic informations which will be used later on
results_fit_5/*  # best-fit parameters, SED and figures.
```
 

### 5. Optionally re-plotting chi2 distribution and compute best-fits
Here we make the SED and chi-square plots, assuming that you have already `sourced` the `SETUP.bash`. 
```
michi2-plot-fitting-results # call it without any argument will print the usage

michi2-plot-fitting-results fit_5.out -flux extracted_flux.txt -source YOUR_SOURCE_NAME
```

Then the output files will be:
```
ls fit_5.pdf          # Yeah, a bunch of best-fit SEDs
ls fit_5.chisq.pdf    # Chi-square histograms
ls best-fit*.txt      # Yeah, fitted parameters
```

One more step: some times the fitted parameters have too small errors. We need to constrain the fitted parameters to not have higher S/N than the photometric data S/N. 




# Usage for LVG Fitting #

## LVG Fitting ##

### 1. Get the code (with git)
Scroll up, there is a green button **"Clone or download"**. Click it and select **"Download ZIP"**. Download it to somewhere, say your working directory `/some/path/`, and uncompress it. 

Currently the ZIP file size is about 52MB. 

Or if you are familiar with git, you can clone it, but note that the repository size is a bit large, like hundreds of MB.  

### 2. Source the code (must under BASH shell)
```
bash
source /some/path/Crab.Toolkit.michi2/SETUP.bash
```

### 3. Prepare line flux data
An example line flux data table, assuming it is named `flux_co_ci.txt`, is like:
```
# X_species  S_species  E_S_species   Molecule
#                                             
101001000  0.21       0.05          CO      
101002001  1.0        0.25          CO      
101004003  1.68       0.1           CO      
101005004  2.2        0.7           CO      
101006005  1.8        0.2           CO      
101007006  2.19       0.184         CO      
102001000  0.70       0.11          C_atom  
102002001  1.72       0.20          C_atom  
```
The first column is a number needed by our fitting, which is unique for each line. The first three digits, 101 means CO, and 102 means C_atom. The second three digits means the upper level, and the third three digits means the lower level. For example CO J=1-0 is 101 001 000, and CO J=9-8 is 101 009 008. For [CI], it's the same, [CI] 3P1-3P0 is 102 001 000, and [CI] 3P2-3P1 is 102 002 001. 

The second column is the integrated flux of the line in units of Jy km/s. And the third column is the error of the line flux. 

The fourth column is optional. You can have more columns as you want, but the fitting code only reads the first three columns.  


### 4. Prepare molecular gas Large-Velocity-Gradient model ####
We also need a molecular gas Large-Velocity-Gradient (LVG) model file before our fitting. Because the Cosmic Microwave Background (CMB) temperature is different at different redshift, such a model file needs to be generated for each redshift. 

You can try to find if there is any corresponding LVG model file under 
`data/lib_LVG/`. It is usually named like `lib_z_1.500_with_CO_and_C_atom_dV_50.lvg`. If you found one, and copy it to your working directory and uncompress it if it is a `*.zip` file.  

Please contact us for a LVG model file.


### 5. Run michi2
If you have already `sourced` the `SETUP.bash`, and have prepared your line flux data file and LVG model file, then just change directory to where the data files are stored and run michi2.

```
cd /path/to/your/data/directory/

ls "flux_co_ci.txt" # make sure you have your line flux file
ls "lib_z_4.055_with_CO_and_C_atom_dV_50.lvg" # make sure you have your LVG model file

michi2-run-fitting-5-components-applying-evolving-qIR # call it without any argument will print the usage

# Now let us do a one-component fit with 2 CPU cores, sampling 15000 chi-squares
michi2_v05 -obs "flux_co_ci.txt" \
           -lib "lib_z_4.055_with_CO_and_C_atom_dV_50.lvg" \
           -out "result/fit.out" \
           -sampling 15000 \
           -parallel 2 \
           | tee log.txt

# Now get the result plots
michi2_plot_LVG_fitting_results.py "result/fit.out"
ls "result/fit.pdf" "result/fit.chisq.pdf" "best-fit_param_"*.txt

# We now also do a two-component fit by inputting "-lib" twice and set a "-constraint" to make sure that one component always has a lower temperature.  
michi2_v05 -obs "flux_co_ci.dat" \
           -lib "lib_z_4.055_with_CO_and_C_atom_dV_50.lvg" \
           -lib "lib_z_4.055_with_CO_and_C_atom_dV_50.lvg" \
           -out "result_two_component_fit/fit.out" \
           -sampling 15000 \
           -parallel 2 \
           -constraint "LIB1_PAR2 < LIB2_PAR2" \
           | tee log_two_component_fit.txt

# Now get the result plots
michi2_plot_LVG_fitting_results.py "result_two_component_fit/fit.out"
ls "result/fit.pdf" "result/fit.chisq.pdf" "best-fit_param_"*.txt
```
If you have enough lines, say more than 4 lines, it is better to use two-component fit. 


### 6. Make a nicer CO CI SLED figure
Copy all the files under the directory `demo/LVG_fitting_michi2/plot_nicer_CO_SLED` except for the `result_two_component_fit` subfolder and `flux_co_ci.txt` and `Plot_nicer_SLED.pdf` files 
to your working directory, then modify the `a_dzliu_code_plot_nicer_SLED.py` code to let it read your own fitting result folder `result_two_component_fit` and your line flux data `flux_co_ci.txt`. The output will be your own version of `Plot_nicer_SLED.pdf`.

