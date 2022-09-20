# Creating updated, scientifically-calibrated mosaics for the RC3 Catalogue
The Third Reference Catalogue of Bright Galaxies (RC3) is a reasonably complete listing of 23,010 large, bright galaxies. Using the latest Sloan Digital Sky Survey’s Data Release 10 (SDSS DR10) data, we provide color composite images and scientifically-calibrated FITS mosaics in all SDSS imaging bands, for all the RC3 galaxies that lie within the survey’s footprint. To get a larger sky coverage, we then conduct the procedures on photographic plates taken by the Digitized Palomar Observatory Sky Survey (DPOSS) for the B, R, IR bands. Due to the positional inaccuracy inherent in the RC3 catalog, the mosaicking program uses a recursive algorithm for positional update first, then conduct the mosaicking procedure using IPAC’s Montage.The program is generalized into a pipeline, which can be easily extended to future survey data or other source catalogs.

## Software Dependencies
- Montage : Stitches images togther
  - AstroPy Montage wrapper
- SExtractor: Source extraction inside the positional update algorithm
- STIFF: Generating color images
- Python v2.7: other versions is fine but minor changes to the code may be necessary
  
  __Packages__
  - NumPy
  - AstroPy preferably v4 dev for the SkyCoord procedures in the source confusion algorithm, but other AstroPy v3.* is fine, I have included the patch as a comment inside rc3.py
  - sqlite3 : generates the searchable database. Not necessary for the mosaicing program to run.
  - Astroquery

Here are some notes that I made on installing these dependencies software on [MacOSx](https://github.com/dorislee0309/workarea-rc3-project/wiki/Installation-on-Factory-Resetted-Mac-OSx-Mavericks)  or on the [NERSC Machines] (https://github.com/ProfessorBrunner/rc3-pipeline/wiki/Installing-dependencies-on-NERSC-Machines)
##Documentation
- [Class Information](http://nbviewer.ipython.org/github/ProfessorBrunner/rc3-pipeline/blob/master/Documentation/Class%20Information.ipynb)
- [SDSS Example](http://nbviewer.ipython.org/github/ProfessorBrunner/rc3-pipeline/blob/master/Documentation/SDSS%20Example.ipynb)
- [DSS Example](http://nbviewer.ipython.org/github/ProfessorBrunner/rc3-pipeline/blob/master/Documentation/DSS%20Example.ipynb)
- [User-defined sub-catalogs](http://nbviewer.ipython.org/github/ProfessorBrunner/rc3-pipeline/blob/master/Documentation/User-defined%20catalog%20%20example.ipynb)
- [Creating your own Survey Class](http://nbviewer.ipython.org/github/ProfessorBrunner/rc3-pipeline/blob/master/Documentation/Creating%20Your%20own%20Survey%20Class.ipynb)
- [Post-processing procedures](http://nbviewer.ipython.org/github/ProfessorBrunner/rc3-pipeline/blob/master/Documentation/Post-processing%20procedures.ipynb)
- [Known Error Conditions](http://nbviewer.ipython.org/github/ProfessorBrunner/rc3-pipeline/blob/master/Documentation/Known%20Error%20Conditions.ipynb)
- [Helpful Tips](http://nbviewer.ipython.org/github/ProfessorBrunner/rc3-pipeline/blob/master/Documentation/Helpful%20Tips.ipynb)
