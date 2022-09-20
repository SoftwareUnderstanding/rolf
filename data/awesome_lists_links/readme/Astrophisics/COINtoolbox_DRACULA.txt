<table>
  <tr >
    <td><img src="https://raw.githubusercontent.com/COINtoolbox/DRACULA/master/images/logo.png"/></td>
    <td align="right"><img src="https://raw.githubusercontent.com/COINtoolbox/DRACULA/master/images/coin.png" width="350"/></td>
</table>

# DRACULA - Dimensionality Reduction And Clustering for Unsupervised Learning in Astronomy

DRACULA is distributed under GPL3 or latter.
It is one of the products of the second edition of the [COIN Residence Program](http://iaacoin.wix.com/crp2015) and is maintained by Michel Aguena (University of Sao Paulo).

If you have any questions, suggestions or just want to be updated about the development of the code, please send an email to coin_dracula+subscribe@googlegroups.com .


If you use DRACULA in your research, please cite [Sasdelli et al, 2015](http://arxiv.org/abs/1512.06810) and [Aguena et al., 2015](http://ascl.net/1512.009).

## Overview

Pipeline to use all methods of data reduction and clustering.
Some cluster quality methods were also implemented.
So far we have implemented:

* For dimensionality reduction:
	* PCA
	* empca
    * kernel PCA
    * isomap
	* DeepLearning 
    * SOM
* For clustering:
	* MeanShift
	* KMeans
	* AffinityPropagation
	* AgglomerativeClustering
	* DBSCAN
* For cluster quality testing:
	* silhouette_index
	* Dunn_index
	* DavisBouldin_index
	* vrc

## Requirements
To run the basic features of this pipeline, you will need:

[numpy](http://www.numpy.org/)

[matplotlib](http://matplotlib.org/)

[sklearn](http://scikit-learn.org/stable/)

Deep Learning requires:

[R](https://www.r-project.org/)

[rpy2](http://rpy2.readthedocs.org/en/version_2.7.x/)

[h2o for R](http://h2o-release.s3.amazonaws.com/h2o/rel-lambert/5/docs-website/Ruser/Rinstall.html)

Additional packages are necessary for:

[EMPCA](https://github.com/sbailey/empca)

[SOM](https://github.com/JustGlowing/minisom)


## Basic use
The idea of the code is to get the function of the pipeline and run the code in a outside dir.
You should first prepare your environment with two simple steps.


There are two approaches for using the pipeline
* Normal Use:

	In this way all the settings of the pipeline are set by a configuration file (`config.py`).
	It only produces results for one specific configuration in each run.
	See the  HOW_TO_USE_CONFIG file to see the possibilities of the configuration file.
* SOM visualisation:

	In this module, the SOM method is used.
	It should be used for visualization of the possible clustering of the data.
	This use does not folow the steps of the pipeline.
	Be careful for this module may take a realy long time according to the parameters chosen.
	The configuration file is `config_som.py`.
	See the README_COMPARISON file for more information.
* Comparison Use:

	It is  strongly recomended that you learn how to use the pipeline in the Normal approach before using this.
	This new module was introduced to run the pipeline variating one of the set of parameters and comparing the results.
	Here a similar configuration file (`config_comparison.py`) is used but with extra keys.
	See the README_COMPARISON file for more information.


### Prepare environment
It is very easy to prepare your environment to run the pipeline.
It can be done in 2 steps:

1.Get the nice functions we prepared. In the pipeline dir, do the command:

	source SOURCE_ME

Alternatively, you can enable your computer to always have access to the DRACULA functions by adding the following line to your `.bashrc` file: 

	export PATH=$PATH:./:'path_to_folder_CODE/'

2.Create your own dir (preferably outside the pipeline dir) to run the code and copy the config.py file there:

	mkdir your_dir
	cd your_dir
	cp PIPELINE_DIR/example_configs/config.py config.py

Now you are ready to run the pipeline functions!

### Pipeline function
Inside your own working dir with the config.py file you can use any of these functions:

To run the whole pipeline (dimensionality reduction, clustering, quality tests and ploting) execute:

	DRAC_ALL

To run just the reduction part execute:

	DRAC_REDUCTION

To run just the clustering execute:

	DRAC_CLUSTERING

To run just the clustering quality execute:

	DRAC_QUALITY

To run just the plotting execute:

	DRAC_PLOT

To run just the plotting of the spectra by gorups execute:

	DRAC_PLOT_SPECS

## Outputs
The outputs of reduction methods are placed in `red_data/`.
They will be input for clustering and plotting unless stated otherwise.

The outputs of clustering methods are placed in `cl_data/`.
They will be input for plotting unless stated otherwise.

The outputs of plotting are placed in `plots/`.

All modules also print the information used and resulting in `info/`.


## Advanced plotting of results
The default plot of the results are in a figure (.png) with all the PCs colored according to the clusters.
If you want to change the extension of the figure or the color arrangement,
change the parameters in the config file (see HOW_TO_USE_CONFIG).

If you want other options, there are a few available using keys in the terminal when executing the `PLOT` command.
Here are the plossibilities:

	-nd	(or --no_diag	) : do not plot diagonal
	-nf	(or --no_fit	) : do not fit in all dimensions simultanniously
	-nc	(or --no_colors	) : do not use colors
	-nl	(or --no_label	) : do not plot label
	-w	(or --window	) : keep plot in interactive window, this will not save the output automaticaly
	-pp	(or --plot_pars	) : plot specified pars, takes a string as input (ex: "1 2")
	-hs	(or --horiz_space) : set horizontal spacing between the plots
	-vs	(or --vert_space) : set vertical spacing between the plots

You can also see them by executing

	DRAC_PLOT -h


## Advanced plotting of spectra
A plot of the orginial spectra by groups is also produced in a figure (.pdf).
The main figure shows the mean value of the spectra of each cluster found.

If you add the key -a , it will also produce a individual plot for each cluster with all the spectra and the mean value.
In the default option of extension (.pdf), all plots are grouped in a single file with multiple pages.
In other cases a extra file for each cluster will be created.

The parameters for plotting the spectra are:

	-w	(or --window	) : keep plot in interactive window, this will not save the output automaticaly
	-a	(or --all_spec	) : plot all spectra

You can also see them by executing

	DRAC_PLOT_SPECS -h
