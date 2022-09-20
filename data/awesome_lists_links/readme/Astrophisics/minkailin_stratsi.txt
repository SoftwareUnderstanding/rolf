# **Stratified and vertically-shearing streaming instabilities**

## Paper
* [`Lin (2021)`](https://ui.adsabs.harvard.edu/abs/2020arXiv201112300L/abstract)

## Requirements
* [`DEDALUS`](https://dedalus-project.org/) (pip v2.2006, works with github commit c8337f3)
* [`EIGENTOOLS`](https://github.com/DedalusProject/eigentools) (v1.2012.1)

## One-fluid model

_Code_  
`stratsi_1fluid.py`:
**Complete code for solving the one-fluid linearized equations.**  

_Physical parameters and options_  
`alpha0`: viscosity parameter to determine diffusion coefficient  
`st0`: Stokes number, assumed constant  
`dg0`: midplane dust-to-gas ratio  
`metal`: metallicity  
`eta_hat`: radial pressure gradient   

`fix_metal`: `True` to set `dg0` from `metal`; `False` to set `dg0` directly          
`tstop`: `True` for finite dust-gas drag; `False` for perfect coupling  
`diffusion`: `True` to include dust diffusion in linearized equations

`kx_min`, `kx_max`, `nkx`: range and sampling in Kx space  

_Numerical parameters_  
`zmin`, `zmax`, `nz_wave`: domain and resolution    
`all_solve_dens`: `True` to solve for all eigenvalues for all Kx  
`first_solve_dens`: `True` to solve for all eigenvalues for first Kx  
`Neig`: number of eigenvalues to find using sparse solver   
`eigen_trial`: trial eigenvalue for sparse solver   
`growth_filter`: upper limit on growth rates

_Outputs_  
`stratsi_1fluid_modes.h5`: eigenvalues and eigenvectors for each Kx  
`stratsi_1fluid_eqm.png`: plot of equilibrium disk structure  
`stratsi_1fluid_vshear.png`: plot of vertical shear rate and dusty buoyancy frequency  

## Two-fluid model

_Codes_  
`stratsi_params.py`: **parameters and analytic vertical structure**  
`stratsi_eqm.py`: **solver for equilibrium horizontal velocity profiles**  
`stratsi_pert.py`: **solver for linearized equations**

_Physical parameter and options_  
`alpha`: alpha viscosity parameter  
`eta_hat`: radial pressure gradient

`dg0`: midplane dust-to-gas ratio  
`metal`: metallicity  
`stokes`: Stokes number

`fix_metal`: `True` to set `dg0` from `metal`; `False` to set `dg0` directly  
`viscosity_eqm`, `viscosity_pert`: include viscous terms in equilibrium and linearized equations?  
`diffusion`: include dust diffusion?  
`backreaction`: include dust feedback onto the gas?

`kx_min`, `kx_max`, `nkx`: range and sampling in Kx space  

_Numerical parameters_  
`zmin`, `zmax`, `nz_wave`: domain and resolution    
`all_solve_dens`: `True` to solve for all eigenvalues for all Kx  
`first_solve_dens`: `True` to solve for all eigenvalues for first Kx  
`Neig`: number of eigenvalues to find using sparse solver   
`eigen_trial`: trial eigenvalue for sparse solver   
`growth_filter`: upper limit on growth rates

_Outputs_  
`stratsi_modes.h5`: eigenvalues and eigenvectors for each Kx    
`eqm_horiz.h5`: equilibrium horizontal velocity profiles   
`stratsi_eqm.png`: plot of equilibrium disk structure   
`stratsi_eqm_drift.png`: plot of dust-gas relative drift  
`stratsi_eqm_cen.png`: plot of dust-gas center-of-mass horizontal velocities  
`stratsi_eqm_epsilon`: single plot of equilibrium dust-to-gas ratio  

## Generic plotting

_Code_  
`stratsi_plot.py`: main plotting tool. Compares one- and two-fluid results.

_Usage examples_  
`python3.7 stratsi_plot.py --mode 8`: plot the 8th mode in Kx space    
`python3.7 stratsi_plot.py --mode 8 --sig 0.5 1` plot the 8th mode in Kx space with growth rate and frequency closest to s=0.5 and omega=1, respectively  
`python3.7 stratsi_plot.py --kx 200`: plot the mode closest to Kx=200    

_Outputs (*.png)_  
`stratsi_plot_growth`: growth rates and frequencies as a function of Kx      
`stratsi_plot_growth_max`: max. growth rate and frequency as function of Kx  
`stratsi_plot_eigen`: all growth rates and frequencies for a single Kx  
`stratsi_plot_eigenfunc`: eigenfunctions  
`stratsi_plot_eigenf2D`: flow visualization in meridional plane (using two-fluid results)    
`stratsi_plot_energy1f`: pseudo-energy decomposition for a single Kx, as a function of z, based one-fluid results  
`stratsi_plot_energy1f_int`: vertically-integrated pseudo-energies as a function of Kx    
`stratsi_plot_energy2f`, `stratsi_plot_energy2f_int`: as above but using two-fluid results  

## Utilities
* `run_problem.sh`  
For running the complete two-fluid problem (computing equilibrium then solving the linearized equations).
* `eigenproblem.py` (OBSOLETE, now using the Python package)\
Copied from the [`EIGENTOOLS`](https://github.com/DedalusProject/eigentools) package, tweaked to allow variable tolerance  (`tol`,`tol_eigen`)
* `stratsi_maxvshear.py`  
For computing the largest vertical shear rate in the disk and its location.
* `stratsi_plot_eqm.py`  
For checking the two-fluid equilibrium horizontal velocity profiles by comparing the right-hand-side and left-hand-sides of the equilibrium equations.

## Special plotting
* `stratsi_plot_visc.py`  
Compares viscous results to inviscid results. Require inviscid results under folder "novisc".
* compare_etas/`stratsi_compare_eta.py`  
Compare results from two different eta values.
* compare_stokes/`stratsi_compare_stokes.py`  
Compare results from two different stokes numbers.
* compare_Z/`stratsi_compare_Z.py`  
Compare results from two different metallicities.
