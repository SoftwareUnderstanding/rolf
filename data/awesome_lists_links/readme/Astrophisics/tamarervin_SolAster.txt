# SolAster

Pipeline to independently derive 'Sun-as-a-star' radial velocity variations using data from the Helioseismic and
Magnetic Imager aboard the Solar Dynamic Observatory.  

![mkdocs/img/rv_animation.gif](mkdocs/docs/img/rv_animation.gif)  

# Documentation

**Documentation Site:**  https://tamarervin.github.io/SolAster/

# Build conda environment

* update dependencies in conda_env.yml [file](conda_env.yml)   
* run the following from the folder containing the .yml file
    * ``conda env create -f conda_env.yml``  
* to add new dependencies, update conda_env.yml [file](conda_env.yml)  
* run the following from the folder containing the .yml file  
    * ``conda env update -f conda_env.yml``
    
# Usage

Pipeline can either be used via scripts located in the [examples](ttps://github.com/tamarervin/SolAster/tree/main/SolAster/examples) 
folder or via installation. Additional instructions can be found in the documentation
site.  

# Installation

* package installation using pip  
* install pip  
* install package   
``pip install SolAster`` 
  
# References  

* Ervin et al. (2021) - Accepted  
* [Milbourne et al. (2019)](https://doi.org/10.3847/1538-4357/ab064a)  
* [Haywood et al. (2016)](https://doi.org/10.1093/mnras/stw187)  
* Based on a technique developed by [Meunier, Lagrange & Desort (2010)](https://doi.org/10.1051/0004-6361/200913551) 
  for SoHO/MDI images.  


# Setup Instructions  

Using our examples requires user specified parameters all of which can be updated in the [settings](https://github.com/tamarervin/SolAster/tree/main/SolAster/tools/settings.py) file.

Update the [Inputs](https://github.com/tamarervin/SolAster/tree/main/SolAster/tools/settings.py) class to include the correct date
frame, cadence, and instrument for your calculations.  

* csv_name: name of the CSV file to store calculations (str)  
* inst: name of instrument to use for calcuations of RV model (str: either NEID' or 'HARPS-N')  
* cadence: querying cadence in seconds (int)  
* start_date: start date for calculations (datetime)  
* end_date: end date for calculations (datetime)  
* diagnostic_plots: whether you would like to plot diagnostic plots (bool)  
* save_fig: path to save figures if diagnostic_plots is True (str)  

Additionally, users can update paths to store CSV files in settings. The current paths are setup to save directly to the downloaded
repository but can be changed for different systems.  

# Outputted results

Our package produces a CSV or pickle file which includes calculation results of velocity components, model RV variations, and 
various solar observables. These results are stored as follows.

* date_obs: calculation time in UT (str)  
* date_jd: calculation time in JD (float)  
* rv_model: model RV variation [m/s]  
* v_quiet: quiet-Sun velocity [m/s]  
* v_disc: velocity of full solar disk [m/s]  
* v_phot: photometric velocity component [m/s]  
* v_conv: convective velocity component [m/s]  
* f_bright: filling factor due to bright regions [%]  
* f_spot: filling factor due to spots [%]  
* f: filling factor [%]  
* Bobs: unsigned magnetic flux [G]  
* vphot_bright: photometric velocity component due to bright regions [m/s]  
* vphot_spot: photometric velocity component due to spots [m/s]  
* f_small: filling factor due to small regions [%]  
* f_large: filling factor due to large regions [%]  
* f_network: filling factor due to network regions [%]  
* f_plage: filling factor due to plage regions [%]  
* quiet_flux: magnetic flux due to quiet-Sun regions [G]  
* ar_flux: magnetic flux due to active Sun regions [G]  
* conv_flux: magnetic flux due to large active regions [G]  
* pol_flux: polarized magnetic flux [G]  
* pol_conv_flux: polarized magnetic flux due to large active regions [G]  
* vconv_quiet: convective velocity component due to quiet-Sun regions [m/s]  
* vconv_large: convective velocity component due to large active regions [m/s]  
* vconv_quiet: convective velocity component due to small active regions [m/s]  


# Examples
Examples are hosted [here](https://github.com/tamarervin/SolAster/tree/main/SolAster/examples):  

1. [Sunpy Example](https://github.com/tamarervin/SolAster/blob/main/SolAster/examples/sunpy_example.ipynb): 
outlines how to use basic Sunpy functions and usages for this package  
   
2. [Component Calculations](https://github.com/tamarervin/SolAster/blob/main/SolAster/examples/component_calculations.ipynb): 
outlines the corrections and component calculation pipeline  
   * creates CSV with calculations of magnetic observables and velocity components  
  
3. [RV Calculations](https://github.com/tamarervin/SolAster/blob/main/SolAster/examples/rv_calculation.ipynb):
outlines calculation of full model RV from velocity components  
   * requires input CSV with velocity components from [example 2](https://github.com/tamarervin/SolAster/blob/main/SolAster/examples/component_calculations.ipynb)  
   * an example CSV file with calculations is stored [here](https://github.com/tamarervin/SolAster/blob/main/SolAster/products/csv_files/calcs/example_calcs.csv)
    
4. [Full Pipeline](https://github.com/tamarervin/SolAster/blob/main/SolAster/examples/full_pipeline.ipynb):
full end-to-end pipeline to calculate 'sun-as-a-star' RVs and magnetic observables 
   
# Issues or Suggestions

* for any issues or bug fixes, please fill out an issue report on the [GitHub page](https://github.com/tamarervin/SolAster/issues)  

# Contact

**Tamar Ervin**: <tamarervin@gmail.com>

**Sam Halverson**: <samuel.halverson@jpl.nasa.gov>
