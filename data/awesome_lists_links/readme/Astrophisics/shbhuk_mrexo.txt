# MRExo

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3306969.svg)](https://doi.org/10.5281/zenodo.3306969)


[MRExo](https://shbhuk.github.io/mrexo/) is a Python script for non-parametric fitting and analysis of the Mass-Radius (M-R) relationship for exoplanets.

We translate [Ning et al. (2018)](https://iopscience.iop.org/article/10.3847/1538-4357/aaeb31)'s `R` script into a publicly available `Python` package called [`MRExo`](http://bit.ly/mrexo_paper). It offers tools for fitting the M-R relationship to a given data
set.  Along with the `MRExo` installation, the fit results from the M dwarf sample dataset from Kanodia et al. (2019) and the Kepler
exoplanet sample from  Ning et al. (2018) are included. 
The code also includes **predicting functions** (M->R, and R->M), and **plotting** functions to generate the plots used in the below preprint.

For detailed description of the code please see [Kanodia et al. (2019)](http://bit.ly/mrexo_paper)


==================


## **Installation**

### Linux/Unix  

1. In the terminal - 
`pip install mrexo`

1. In the terminal - 
 `pip install git+https://github.com/shbhuk/mrexo.git -U`

OR

1. In the terminal - 
`pip install git+ssh://git@github.com/shbhuk/mrexo.git -U `

OR 

1. In the terminal - 
`git clone https://github.com/shbhuk/mrexo`

2. In the repository directory 
`python setup.py install`

The -U is to upgrade the dependencies.


### Windows 

 1. Install Git - [Git for Windows](https://git-for-windows.github.io/)

 2. Install pip - [pip for Windows](https://pip.pypa.io/en/stable/installing/)

 3. In cmd or git cmd - 
  `pip install git+https://github.com/shbhuk/mrexo.git -U`
 
 OR 
 
 `pip install mrexo`
 
 OR 
 
 Clone or Download the zip file and then in the repository directory
 `python setup.py install`
 
To sign up for updates, please join the Google Group linked here - https://groups.google.com/forum/#!forum/mrexo

================== 
 
## **Citation**

Guidelines to cite this package can be found [here](https://github.com/AASJournals/Tutorials/blob/master/Repositories/CitingRepositories.md).

The relevant paper can be found on [ADS](<http://bit.ly/mrexo_paper>).

==================

