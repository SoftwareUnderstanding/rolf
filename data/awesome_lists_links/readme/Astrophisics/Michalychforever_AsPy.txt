# AsPy
This code computes the determinants of aspherical fluctuations on the spherical collapse background. 

There are 3 codes in total:

* det_master.py - computes the determinant in the EdS universe for all orbital numbers except for l=1.

* dipole_IR_master.py - computes the determinant in the dipole sector, which is plagued by IR divergences and requires a special treatement. The calculation is done for the EdS universe.

* LCDM_dipole_IR_master.py - computes the dipole determinant in the LCDM universe.

Additional data files:

Omftab.dat - the file with (\eta, \Omega_m/f^2), where \eta = log D, D is the linear growth factor, f is the logarithmic growth factor = dln D/dln a (a - scale factor),  \Omega_m(\eta) is the time-dependent matter density fraction.

matterpower_horizonRun3.dat - the linear matter power spectrum of our simulations

The codes are (hopefully) self-explaining, if you have problems, please e-mail @ ivanov(at)ias.edu  
The details of the numerical algoritm are given in a supplementary file 'notes.pdf' 
 
If you use this code in your research, please cite the paper arXiv:1811.07913
