# Wigglewave

Wigglewave is a FORTRAN code that uses a finite difference method to solve the linearised governing equations for Torsional Alfv&egrave;n Waves (TAWs) propagating in a plasma with negligible plasma beta and in a force-free axisymmetric magnetic field with no azimuthal component embedded in a high density divergent tube structure. 

Wigglewave is fourth order in time and space using a fourth-order central difference scheme for calculating spatial derivatives and a fourth-order Runge-Kutta (RK4) scheme for updating at each timestep. The solutions calculated are the perturbations to the velocity, v and to the magnetic field, b. All variables are calculated over a uniform grid in radius r and height z. An in-depth explanation of the code can be found in the accompanying PDF document Wigglewave_equations.pdf.

Wigglewave was used extensively in the papers: 

*Enhanced phase mixing of torsional Alfvén waves in stratified and divergent solar coronal structures – Paper I. Linear solutions*, https://academic.oup.com/mnras/article-abstract/510/2/1910/6449388?redirectedFrom=fulltext 

and 

*Enhanced phase mixing of torsional Alfvén waves in stratified and divergent solar coronal structures – II. Non-linear simulations*, https://academic.oup.com/mnras/article-abstract/510/2/2618/6460507?redirectedFrom=fulltext

## Usage

WiggleWave can be run simply compiled and run using gfortran. The code requires no input files but the user is able to change the problem parameters in the Constants module, the parameters are listed in the first table below.

The code outputs include solutions for the velocity perturbation, the magnetic field perturbation and wave envelopes for these perturbations. The outputs are saved as .dat files and are listed in the second table below.

To calculate the total wave energy flux from the wave envelopes the user must first convert the output files from .dat to .sav files using the IDL script makesavs.pro and then use the IDL script wave_enrgy.pro on the .sav files generated.

## Input Parameters

The parameters that can be changed are at the begining of the code. These parameters are:

| Parameter | Description |
| --- | --- |
| B0        | background magnetic field strength                               |
| rho0      | characteristic density                                           |
| nr        | number of cells in radial direction                              |
| nz        | number of cells in vertical direction                            |
| rmin      | minimum radius of domain                                         |
| rmax      | maximum radius of domain                                         |
| zmin      | minimum height of domain                                         |
| zmax      |  maximum height of domain                                        |
| t_end     |  simulation run time                                             |
| t_interval|  time interval between outputs                                   |
| t0        |  rampup time                                                     |
| save_dir  |  directory to save outputs to                                    |
| H      | magnetic scale height                                               |
| visc   | kinematic viscosity                                                 |
| period | wave period in seconds                                              |
| alpha  | &alpha; parameter, defines density scale height through 	&alpha; = H/H<sub>&rho;</sub>    |
| zeta   | density contrast between tube centre and background density         |
| u0     | Alfv&egrave;n wave velocity amplitude in ms<sup>-1</sup>            |
| r0     | radius of central higher density tube                               |
| omega  | the Alfv&egrave;n  wave frequency, currently defined based on the period |
| topdamp  | logical for top boundary damping|
| outdamp  | logical for outer boundary damping |
| restart  | logical for whether to load from a restart file |
| v_in     | file names for the velocity restart file |
| b_in     | file names for the magnetic field restart file|
| restart_time | the simulation time for the restart files |
| last_output  | the output index for the restart files |

## Outputs

All outputs are contain 2D arrays defining some variable at each grid point over the domain.

| Output | Description |
| --- | --- |
| Va.dat           | Alfv&egrave;n velocity  |
| Br.dat           | background magnetic field in radial direction      |
| Bz.dat           | background magnetic field in verical direction      |
| rho.dat          | density |
| phi.dat          | curvilinear coorinate along field lines &phi;     |
| psi.dat          | curvilinear coorinate along magnetic surfaces &psi;      |
| v_n.dat            | velocity perturbation output for output index n  |
| env_v_n.dat        | envelope for velocity perturbation output for output index n   |
| b_n.dat        | magnetic field perturbation output for output index n |
| env_b_n.dat            | envelope for magnetic field perturbation output for output index n  |

## makesavs.pro

makesavs.pro is an IDL script that can be used to transform the outputs from Wigglewave from .dat fils into .sav files which can then be accessed by wave_energy.pro to calculate the wave energy flux across magnetic surfaces. 

The user must specify the grid dimensions in radial and vertical directions, nr and nz; the location of the output files to convert, dir; and the output index of the files to convert, snapshot, which must be written as a three digit number including leading zeroes. 

## wave_energy.pro

wave_energy.pro is an IDL script that uses outputs from Wigglewave to calculate the wave energy flux across magnetic surfaces at different heights. Before using wave_energy.pro, however, the outputs must be converted to .sav format using makesavs.pro. A detailed description of how wave_energy.pro works and what calculations it makes can be found in the accompanying PDF document wave_energy.pdf

## Visualisation scripts

The IDL visulasation scripts that used Wigglewave outputs to generate figures that were used in these two papers can be found in under Visualisation_scripts. The purpose of each script is as follows:

| Script | Description |
| --- | --- |
| plotgraphs.pro | Produces two panel plots comparing the azimuthal velocity and azimuthal magnetic field perturbation between TAWAS and Wigglewave outputs respectively, used to produce Figures 10 and 11 in Paper I.|
| energy_graphs.pro | Produces a plot of the normalised wave energy flux across a given magnetic surface against height for TAWAS and Wigglewave outputs, used to produce Figures 12, 13 and 14 in Paper I.|
| va_graph.pro | Plots a graph of equilibrium density and Alfv&egrave;n speed against radius at the lower boundary and a contour plot of the Alfv&egrave;n speed across the domain, used to produce Figures 1 and 2 in Paper II.|
| vplot_wiggle.pro | Produces a graph of the azimuthal velocity against the radius, r, and height, z, for a specified Wigglewave output, used to produce Figure 3 in Paper II.|
