# mswavef

MSWAVEF - Code to calculate hydrogenic and non-hydrogenic momentum-space electronic wavefunctions is presented.  Such wavefunctions are often required to calculate various collision processes, such as excitation and line broadening cross sections.  The hydrogenic functions are calculated using the standard analytical expressions.  The non-hydrogenic functions are calculated within quantum defect theory according to the method of Hoang Binh and van Regemorter (1997).  Required Hankel transforms have been determined analytically for angular momentum quantum numbers ranging from zero to 13 using Mathematica.  Calculations for higher angular momentum quantum numbers are possible, but slow (since calculated numerically).  The code is written in IDL.

The Mathematica notebook to calculate the Hankel transforms is provided.  Also a file of tests that were performed and zipped versions of the output of some tests.

Bug reports: paul.barklem@physics.uu.se

Please cite the DOI if you use the code in research:
[![DOI](https://zenodo.org/badge/21607/barklem/mswavef.svg)](https://zenodo.org/badge/latestdoi/21607/barklem/mswavef)

Reference:

Hoang Binh, D., and Henri van Regemorter. “Non-Hydrogenic Wavefunctions in Momentum Space.” Journal of Physics B: Atomic 30, no. 1 (1997): 2403–16. doi:10.1088/0953-4075/30/10/014.
