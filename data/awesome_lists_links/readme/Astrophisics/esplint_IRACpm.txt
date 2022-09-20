# IRACpm
IRACpm R Package: Applies a 7-8 order distortion correction to IRAC astrometric data from the Spitzer Space Telescope
and includes a function for measuring apparent proper motions between different Epochs.

#Instructions for installation into R

install.packages("devtools")

library(devtools)

install_github("esplint/IRACpm")

#Basic work flow:

1) Read in files containing output from the Spitzer Science Center’s APEX single frame module
form MOPEX using read.in.data.

2) Measure image central world coordinates and rotations with CD.solver, CD.solver2, CD.solver3,
or CD.solver4

3) Calculate average coordinates for each star of interest with calc.all.ch12

4) Repeat for other epochs

5) Run mucalc to measure apparent proper motions
(If accurate relative astrometry is wanted without proper motions, just follow steps 1-3.)

#Example

Example datasets and output for CD.solver is CD1, 
read.in.data is data1, and input data for mucalc is epochs3.

The measured distortion corrections contain a time varying scale factor
that can either be estimated or measured. In the example below, 
coor.calc converts pixel coordinates (-100, 104) into RA and Dec
using the [3.6] distortion correction (wa_pars1), the [3.6] pixel 
bias corrector (ca_pix1), image central coordinates and rotation 
(55.824647,32.3151426,-4.5420559), and explicitly indicating that 
the calculation is for channel one (1). In the first calculation,
a measured scale factor (1.000083,1.000295) is used. 
In the second, coor.calc estimates the scale factor using the
time of observations as indicated in data1. The difference between
these two calculations is ~2 miliarcseconds.


data(CD1,ca_pix1,wa_pars1,data1)

options(digits=10)

1) using a measured scale factor

coor.calc(ca_pix1,wa_pars1,c(55.824647,32.3151426,-4.5420559),-100,104,c(1.000083,1.000295),1)

2) estimating a scale factor from HMJD.

coor.calc(ca_pix1,wa_pars1,c(55.824647,32.3151426,-4.5420559),-100,104,data1,1)


