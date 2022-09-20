# MAH
Calculate the posterior distribution of the "minimum atmospheric height" (MAH) of an exoplanet by inputting the joint posterior distribution of the mass and radius.

README file by David Kipping, Columbia University, Dept. of Astronomy
(d.kipping at columbia.edu). Please feel free to contact me regarding
questions on MAH or any bugs you may spot.

===================
=== 0. CONTENTS ===
===================

1. What is MAH?
2. Compiling & Executing
3. How to use with your data
4. Outputs
5. Caveats

=======================
=== 1. What is MAH? ===
=======================

Fortran 90 code for calculating the Minimum Atmospheric Height (MAH) of        
an exoplanet. This calculation only requires knowledge of the planet's mass
and radius.

For an inputted array of planetary masses and planetary radii, MAH returns 
R_MAH = the minimum atmospheric height. Please see Kipping, Spiegel & Sasselov 
(2013) for details on how this calculation is made and the applications. If you 
use this code or the equations within, please cite Kipping, Spiegel & Sasselov 
(2013) and Zeng & Sasselov (2013).

================================
=== 2. Compiling & Executing ===
================================

This tarball contains two Fortran 90 files: 
{1} MAH.f90 = the module which computes R_MAH 
{2} MAH_call.f90 = example calling code. 

The code may be compiled together using...
g95 -O3 -c MAH.f90 && g95 -O3 -o MAH MAH_call.f90 MAH.o

and the code can be executed using...
./MAH

or, to dump the screen output into a summary file...
./MAH > summary.txt

There is an example set of data included, which comes from Anglada-Escude et 
al. 2013: http://adsabs.harvard.edu/abs/2013A%26A...551A..48A) for the planet
GJ 1214b, to whom we are grateful for sharing their data. Needless to say if you 
use this data please cite Anglada-Escude et al. 2013. The current example file 
should run in ~2 mins on a typical single core.

====================================
=== 3. How to use with your data ===
====================================

Consider that you have fitted some exoplanet observations with a Monte Carlo
technique, such as MCMC, and have a derived joint posterior distribution of
the planet's mass (MP) and the planet's radius (RP). To use MAH directly
without any modification to code itself, simply export a ASCII file called
"example_data.dat" into the same directory as the MAH code containing
two columns (MP and RP, in Earth units, respectively) and 10^5 rows. If you want
to use a different number of rows, simply change "n" in MAH_call.f90 from 1D5
to whatever you want. Then, simply compile and execute MAH to complete a run.

Advanced users may want to import higher dimensional posteriors directly into
MAH or simply call the MAH.f90 module themselves without the wrapper provided
here and of course you are welcome to do so!

==================
=== 4. Outputs ===
==================

MAH produces two types of output. Firstly, a set of summary statistics are
dumped onto the screen at the end of the calculation. These statistics follow 
the same terminology used in our paper. The numbers are outputted in LaTeX
ready format to copy and paste into your papers. However, depending upon the
number of significant figures and decimal places desired, you may need to
change the formatting rules defined within the MAH_call.f90 code.

The second type of output from MAH are two ASCII files exported to the same
directory in which the code is executed. The file names of these two are:

100_H2O.dat = Output assuming a 100%-water planet for the RH20
75_H2O_25_MgSiO3.dat = Output assuming a 75%-water-25%-silicate planet for RH20

The difference between the two files is typically quite small, but I recommend
using the 75%-water-25%-silicate model, in general.

Each file has three columns and the same number of rows as in the original
inputted array. The i'th row of the output files corresponds to the i'th row of
the input file. The columns are:

{1} R_H20 = Radius of the planet, in Earth radii, assuming a water composition
{2} R_MAH = Minimum atmospheric height of the planet, in Earth radii
{3} valid = Logical flag (T or F) dictating whether this row is trustable or not

The first two columns are self-explanatory to anyone who has read our paper.
The last column is less obvious and essentially checks whether the inputted
planetary mass falls within the regime covered by the Zeng & Sasselov (2013)
models. If the inputted mass is extreemly small or large, the models become
unreliable and so we flag such cases in this column. All summary statistics
calculated in MAH_call.f90 know about this and exclude such rows in their
calculations.

A typical use of these outputs would be to create a histogram of the second
column of the file 75_H2O_25_HgSiO3.dat, which would represent the posterior
distribution of the minimum atmospheric height of the planet (in units of
Earth radii). One could also divide this column by the inputted RP to get
(R_MAH/R_P) i.e. a relative measure of R_MAH.

==================
=== 5. Caveats ===
==================

* It is very important that the inputted masses and radii are in units of
Earth masses and Earth radii respectively.

* It is also important that users understand that the following statement is
false:
Probability of planet being atmosphere-less = 1 - P(RMAH>0)

* In fact if your posterior distribution of RMAH peaks to negative values, then 
the MAH method says *nothing* about whether the planet does or does not have an
atmosphere. Therefore, please do not interpret such a result as indicating a
rocky planet because this is fundamentally wrong! However, if P(RMAH>0) is
high, then you may interpret this as indicating that the planet has an extended
atmosphere.
