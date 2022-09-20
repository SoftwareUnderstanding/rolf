# TAWAS

TAWAS is an IDL script for solving the wave equation for Torsional Alfv&egrave;n Waves (TAWs) propagating in a viscous plasma. The background magnetic field is axisymmetric and force-free with no azimuthal component, and the plasma beta is assumed to be negligible. The solution is calculated over a uniform numerical grid with coordinates r and z for the radius and height respectively. An in-depth explanation of the calculations made in this code can be found in the accompanying PDF document TAWAS Calculations.

The solution is calculated using the analytical formula presented in the paper:

*Enhanced phase mixing of torsional Alfvén waves in stratified and divergent solar coronal structures – Paper I. Linear solutions*, https://academic.oup.com/mnras/article-abstract/510/2/1910/6449388?redirectedFrom=fulltext 

This paper makes extensive use of the TAWAS code for considering the role of wave reflection, running paprameters studies and for comparing the analytical solutions to exact solutions of the linearised governing equations, calculated using the Wigglewave code.

## Usage

TAWAS can be run using IDL. The code requires no input files but the user is able to change the problem parameters at the top of the code, the parameters are listed in the first table below. Under the parameters the user can set the variable 'plotting' which decides which plots are shown by the script. The options for plotting are given in the code. The user can chose the location to save the outputs by changing the variables save_dir and run_name listed just below the parameters.

The code outputs include solutions for the velocity perturbation, the magnetic field perturbation and the wave energy flux . The outputs are saved as .sav files and are listed in the second table below.

## Constraints

The r and z dimensions of the domain, rdim and zdim, must be constrained according to the value of the magnetic scale height, H. The radial dimension rdim must not exceed 2.4048 H. At this point the code encounters a singularity as the direction of the magnetic field lines turns back in towards the lower boundary.

Furthermore if zdim is large then the magnetic surfaces over which wave energy flux is calculated may not fit in the radial extent of the domain and the calculations will be incorrect. The user must check manually whether the magnetic surfaces fit within the domain by turning plotting to 1 or 4 and looking at IDL window 5 which plots the magnetic surfaces.

## Input Parameters

The parameters that can be changed are at the begining of the code. These parameters are:

| Parameter | Description |
| --- | --- |
| b0     | background magnetic field strength                               |
| rho0   | characteristic density                                           |
| rsize  | number of cells in radial direction                              |
| zsize  | number of cells in vertical direction                            |
| rdim   | dimension of domain in the radial direction                      |
| zdim   | dimension of domain in the vertical direction                    |
| H      | magnetic scale height                                            |
| alpha  | &alpha; parameter, defines density scale height through 	&alpha; = H/H<sub>&rho;</sub>    |
| r0     | radius of central higher density tube                            |
| zeta   | density contrast between tube centre and background density      |
| u0     | Alfv&egrave;n wave velocity amplitude in ms<sup>&-1;</sup>       |
| period | wave period in seconds                                           |
| t0     | wave phase of solution as time in seconds (usually left as zero) |
| viscosity | kinematic viscosity |


## Outputs

| Output | Format | Description |
| --- | ----- | --- |
| br           | 2D array              | background magnetic field in radial direction        |
| bz           | 2D array              | background magnetic field in vertical direction      |
| rho          | 2D array              | density across domain                                |
| phi          | 2D array              | curvilinear coorinate along field lines &phi;        |
| psi          | 2D array              | curvilinear coorinate along magnetic surfaces &psi;        |
| hparam       | 2D array              | phase parameter h , see below for more details       |
| W            | 2D complex array      | wave parameter W , see below for more details        |
| v_env        | 2D complex array      | envelope of the velocity perturbation                |
| b_env        | 2D complex array      | envelope of the magnetic field perturbation          |
| v            | 2D complex array      | azimuthal velocity perturbation                      |
| bwave        | 2D complex array      | azimuthal magnetic field perturbation                |
| sigma        | 2D array              | WKB parameter &sigma;, see below for more details    |
| zscale       | 1D array              | heights at which each magnetic surface intersects the z-axis, in metres       |
| hscale       | 1D array              | heights at which each magnetic surface intersects the z-axis, in units of H   |
| en_lvl       | 1D array              | wave energy flux across each magnetic surface in Watts                        |
| en_lvl_norm  | 1D array              | wave energy flux normalised by the wave energy flux at the lowest surface     |

## norefl and uncorr

These scripts are modifications of TAWAS with the variable 'diss' changed to either: ignore the effect of reflecion, in the case of norefl; or to use an uncorrected form, &Upsilon; + R instead of  &Upsilon; - iR,  in the case of uncorr.

## Visualisation scripts

The IDL visulasation scripts that used TAWAS outputs to generate figures for the paper, *Enhanced phase mixing of torsional Alfvén waves in stratified and divergent solar coronal structures – Paper I. Linear solutions*, can be found in under Visualisation_scripts. The purpose of each script is as follows:

| Script | Description |
| --- | --- |
| plotgraphs.pro | A coloured contour of the density is plotted against the radius, r, and height, z,  A contour line is overplotted to show the position of the tube boundary, used to produce Figure 2. |
| solution_examples.pro | Coloured contours of both the azimuthal velocity and magnetic field perturbations are plotted againt the radius, z, and height, z. Additionally a line plot of the normalised wave energy flux is plotted against height,used to produce Figures 3, 4 and 5. |
| results.pro | The normalised wave energy flux is plotted against height for six different sets of parameters, on each graph the wave energy flux is plot using the outputs of norefl.pro, uncorr.pro and tawas.pro, used to produce Figures 6 and 7.| 
| scale_grid.pro | A colourful contour of the normalised wave energy flux, across the highest magnetic surface, is plotted against magnetic and density scale heights, used to produce Figure 8.|
| visc_period_grid.pro | A colourful contour of the normalised wave energy flux, across the highest magnetic surface, is plotted against viscosity and wave period, used to produce Figure 9.|
