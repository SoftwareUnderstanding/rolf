# abo-cross

Hydrogen broadening Cross-Section Calculator

This code was announced in 
  - Barklem, P S, S. D. Anstee, and B J O’Mara. “Line Broadening Cross Sections for the Broadening of Transitions of Neutral Atoms by Collisions with Neutral Hydrogen.” Publications Astronomical Society of Australia 15 (1998): 336–38.

If you use this code, please cite this paper and the other papers on which it is based: 

  - Anstee, S. D., and B J O’Mara. “Width Cross-Sections for Collisional Broadening of s-p and p-s Transitions by Atomic Hydrogen.” Monthly Notices of the Royal Astronomical Society 276, no. 3 (1995): 859–66.
  - Barklem, P S, and B J O’Mara. “The Broadening of p-d and d-p Transitions by Collisions with Neutral Hydrogen Atoms.” Monthly Notices of the Royal Astronomical Society 290, no. 1 (1997): 102–6.
  - Barklem, P S, B J O’Mara, and J. E. Ross. “The Broadening of d-f and f-d Transitions by Collisions with Neutral Hydrogen Atoms.” Monthly Notices of the Royal Astronomical Society 296, no. 4 (1998): 1057–60. doi:10.1046/j.1365-8711.1998.01484.x.

Please also cite the DOI:
[![DOI](https://zenodo.org/badge/21607/barklem/abo-cross.svg)](https://zenodo.org/badge/latestdoi/21607/barklem/abo-cross)


This program interpolates cross-sections from the Anstee, Barklem and O'Mara computed tables of line broadening parameters for transitions of neutral atoms broadened by collisions with neutral hydrogen atoms.  The code will interpolate from the tables, and compute the line width at a given temperature. It was written with the idea that users would take the code and merge into their own codes, but it can be used in stand alone mode. Another aim to make available electronic verisons of the tables for convenience.

I provide the original basic version of the program (v1.1) and a newer verison (v2.0).  The main internal codes are exactly the same but a different front end routine (plus necessary extras) is provided in v2.0 which has the following advantages:

 - The computation of effective principal quantum number is done by the program, input is simply energy levels;
 - Uses input and output files which has several advantages:
   - easier computation for long line lists;
   - record is automatically kept of all calculations;
 - Atomic masses of elements are stored in the program.
 
Each version has its own README.

Bug reports: paul.barklem@physics.uu.se
