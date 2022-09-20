# phase_space_cosmo_fisher
Codes used to produce Fisher matrix 2D contours found in Stark, Miller Huterer 2017 https://arxiv.org/abs/1611.06886 (v2)

----------------------------
MAIN codes

*fisher_matrix.py* 
This is the main code which calculates the fisher matrix and generates 2d marginalized contours of cosmological parameters given a specified redshift array and cosmological case. See comments in code. You can also plot the derivatives used in the fisher matrix through the plot_derivatives function.

*functions_fisher.py* 
Functions such as the escape velocity profile and cosmological quantities used by fisher_matrix.py are housed here.

*paper_figures.py*
Functions to generate the specific figures found in  Stark, Miller, Huterer 2017 paper are stored here.

----------------------------
OTHER codes
*einasto_error.py*
maps NFW errors to Einasto parameter errors for a given fiducial cluster.

*mcmc_sigma_cosmo.py*
Utilizes the ClusterEnsembles package (https://pypi.python.org/pypi/Cluster_Ensembles/1.16) to generate projected mass density functions as a function of cosmology. Uses emcee routine to sample posterior. See Appendix of our paper. This is the code we use to generate the likelihoods in Fig. 13.

*3dplot_param.py*
Generates 3D plots of qH^2 and other cosmological quantities as a function of redshift and cosmology.


----------------------------
TUTORIAL

Open fisher_matrix.py and specify the number of clusters and redshift range (see 'sample_redshift_array' and 'sample_number_of_clusters') used to generate 2D contours for a given cosmological case. Then specify which of the three parameter cases you want to constraint (pick either 'flat', 'w_z', 'non_flat', or others... see code). Run fisher_matrix to generate 2d contours of whatever cosmological case you specify. You can also read in-code documentation and Stark et al 2017 (https://arxiv.org/abs/1611.06886) for more detals.

----------------------------
LIBRARIES 
You will need cosmolopy, astropy, sympy, corner and emcee.

----------------------------
If you have any questions you can email the author: alejo@umich.edu
