cluster-in-a-box
================

Statistical model of sub-millimeter emission from embedded protostellar clusters. The paper describing the model is <a href="https://ui.adsabs.harvard.edu/abs/2015ApJ...807L..25K/abstract">Kristensen & Bergin 2015, ApJL</a>. The model is written in python and uses astropy. 

If using this model, please cite the DOI along with the paper: <br>
<a href="http://dx.doi.org/10.5281/zenodo.13184"><img src="https://zenodo.org/badge/doi/10.5281/zenodo.13184.svg" alt="10.5281/zenodo.13184"></a><br>
<a href="https://ascl.net/1610.008"><img src="https://img.shields.io/badge/ascl-1610.008-blue.svg?colorB=262255" alt="ascl:1610.008" /></a>

The model consists of three modules grouped in two scripts, both stored under model. The first (cluster_distribution) generates the cluster based on the number of stars, input initial mass function, spatial distribution and age distribution. The second (cluster_emission) takes an input file of observations, determines the mass-intensity correlation and generates outflow emission for all low-mass Class 0 and I sources. The output is stored as a FITS image where the flux density is determined by the desired resolution, pixel scale and cluster distance. 

The observational data used for generating the model results shown in the paper are stored under observations. The data consist of single-dish JCMT data of methanol obtained with the HARP instrument at 338.4 GHz. Only the text file properties.dat is required for running the cluster_emission script. 

Future updates to be implemented:
* Implement an evolutionary module; currently the number of stars and age distribution are input but sometimes a cloud mass and lifetime are desirable instead, with the appropriate star formation efficiency as a free parameter. 
* Implement other transitions and species, in particular water and high-J CO as based on Herschel observations. 
* Implement velocity distribution and generate (position, position, velocity) cubes rather than just images of velocity-integrated intensity. 
