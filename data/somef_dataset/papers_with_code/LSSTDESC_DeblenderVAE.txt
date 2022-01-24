# Deblender_VAE

This repository hosts the code used in Arcelin, B., Doux, C., Aubourg, E. & Roucelle, C. _Deblending galaxies with Variational Autoencoders: a joint multi-band, multi-instrument approach_. Mon. Not. R. Astron. Soc. (2020). https://arxiv.org/abs/2005.12039.

This paper presents a method to deblend (i.e. isolate) overlapping galaxies in multiband astronomical survey images. The focus is on the LSST weak lensing survey and potential improvements from including images from ESA's Euclid satellite.

In brief, the method uses two networks:
- a variational autoencoder (Kingma 2014, https://arxiv.org/abs/1312.6114) to denoise isolated galaxy images.
- and another network, which has the same architecture as the VAE, to deblend the galaxies. In this network, only the encoder is trained, since the decoder is fixed: weights are fixed from those of the VAE's decoder.

This folder contains the scripts for the images generation, the VAE and deblender training and the differents plots and tests.

The images are generated with GalSim (https://github.com/GalSim-developers/GalSim, doc: http://galsim-developers.github.io/GalSim/_build/html/index.html) from parametric models fitted to real galaxies from the HST COSMOS catalog (which can be found from here: https://github.com/GalSim-developers/GalSim/wiki/RealGalaxy%20Data).

The list of released versions of this package can be found [here](https://github.com/LSSTDESC/DeblenderVAE/releases), with the master branch including the most recent (non-released) development.

## Installation
1. Clone the repository
```
git clone https://github.com/LSSTDESC/DeblenderVAE.git
cd DeblenderVAE
```
2. Install 
- with [conda](https://www.anaconda.com/products/individual) or [miniconda](https://docs.conda.io/en/latest/miniconda.html)
  - if you don't want to use GPU
    ```
    conda env create -f ressources/env_TF.yml
    conda activate env_vae_tensorflow
    ```
  - if you want to use GPU
    ```
    conda env create -f ressources/env_TF_gpu.yml
    conda activate env_vae_tensorflow_gpu
    ```


## Required packages
- scipy
- numpy
- jupyter
- jupyterlab
- matplotlib
- astropy
- keras
- tensorflow=1.13.1
- tensorflow-probability=0.6.0
- galsim
- seaborn
- pandas
- tqdm
