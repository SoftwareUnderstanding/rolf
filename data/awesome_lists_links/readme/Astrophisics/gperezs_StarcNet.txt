
<h1 align="center">
  <br>
  <a><img width="300" src="logo.png" alt="StarcNet"></a>
</h1>

<h4 align="center">Machine Learning for Star Cluster Classification</h4>

<p align="center">
    <a href="https://github.com/gperezs/StarcNet/commits/master">
    <img src="https://img.shields.io/github/last-commit/ArmynC/ArminC-AutoExec.svg?style=flat-square&logo=github&logoColor=white"
         alt="GitHub last commit">
    <a href="https://github.com/gperezs/StarcNet/issues">
    <img src="https://img.shields.io/github/issues-raw/ArmynC/ArminC-AutoExec.svg?style=flat-square&logo=github&logoColor=white"
         alt="GitHub issues">
    <a href="https://github.com/gperezs/StarcNet/pulls">
    <img src="https://img.shields.io/github/issues-pr-raw/ArmynC/ArminC-AutoExec.svg?style=flat-square&logo=github&logoColor=white"
         alt="GitHub pull requests"></a>
</p>


PyTorch code for classification of star clusters from galaxy images
taken by the Hubble Space Telescope (HST) using StarcNet. 
StarcNet is a convolutional neural network (CNN) trained to classify
5-band galaxy images into four morphological classes. 
The target galaxies used in this project are provided by the [Legacy
ExtraGalactic UV Survey
(LEGUS)](https://archive.stsci.edu/prepds/legus/).
The running time of StarcNet in a Galaxy of 3,000
objects is about 4 mins on a CPU (4 secs with a GPU).

![title_image](title_im.jpg)


## Table of contents
* [Installing / Getting started](#installing-/-getting-started)
	* [Using Anaconda](#using-anaconda)
	* [Using Virtualenv](#using-virtualenv)
* [Run StarcNet](#run-starcnet)
	* [Using local data](#run-starcnet-with-local-data)
	* [Using LEGUS catalogs](#run-starcnet-with-online-legus-catalogs)
* [Acknowledgements](#acknowledgements)


## Installing / Getting started

1. **Clone the repository:** To download this repository run:
```
$ git clone https://github.com/gperezs/StarcNet.git
$ cd StarcNet
```

In the following sections we show two ways to setup StarcNet. Use the one that suits you best: 
* [Using virtualenv](#using-virtualenv)
* [Using Anaconda](#using-anaconda)

### Using virtualenv

2. **Install virtualenv:** To install virtualenv run after installing pip:

```
$ sudo pip3 install virtualenv 
```

3. **Virtualenv  environment:** To set up and activate the virtual environment,
run:
```
$ virtualenv -p /usr/bin/python3 venv3
$ source venv3/bin/activate
```

To install requirements, run:
```
$ pip install -r requirements.txt
```

4. **PyTorch:** To install pytorch run:
```
$ pip install torch torchvision
```

-------
### Using Anaconda

2. **Install Anaconda:** We recommend using the free [Anaconda Python
distribution](https://www.anaconda.com/download/), which provides an
easy way for you to handle package dependencies. Please be sure to
download the Python 3 version.

3. **Anaconda virtual environment:** To set up and activate the virtual environment,
run:
```
$ conda create -n starcnet python=3.*
$ source activate starcnet
```

To install requirements, run:
```
$ conda install --yes --file requirements.txt
```

4. **PyTorch:** To install pytorch follow the instructions [here](https://pytorch.org/).

-------
## Run StarcNet

StarcNet will classify objects from a single galaxy or a list of galaxies. 
Galaxies to be classified should be added into `targets.txt`. 
StarcNet runs using mosaics (`.fits` files with the galaxy photometric information) 
and catalogs (`.tab` files with object coordinates) saved locally in 
`legus/frc_fits_files/` and `legus/tab_files/` respectively.
StarcNet includes the option to also download the galaxy mosaics from a single `.tar.gz` 
file per galaxy as in LEGUS. See next two sections to run StarcNet with and without downloading mosaics.
 
This repository comes ready to classify objects from NGC1566 with the option of downloading 
the mosaics (See `target.txt` and `frc_fits_links.txt`). StarcNet predictions of all galaxies in `targets.txt` 
are saved into `output/predictions.csv`. In addition to `output/predictions.csv`, StarcNet saves the predictions 
with the classification scores of each independent galaxy into a separate `.tab` file `output/<galaxy name>.tab`.

To run StarcNet on NGC1566:

```
$ bash run_starcnet.sh 1
```

To produce visualization of the predictions over the galaxy image run: 
```
$ python src/run_visualization.py
```
The visualization script will create an image per galaxy previously classified 
(i.e. a visualization of each galaxy in `output/predictions.csv`). 
Each visualization output is saved into `output/visualizations/<galaxy name>_predictions.png`

Seei also NGC1566 [demo](demo.ipynb) ipython notebook file.

### Run StarcNet with local data

1. Save the 5 mosaic's `.FITS` files of each galaxy into `legus/frc_fits_files/` folder.
2. Save catalog `.tab` file of each galaxy into `legus/tab_files/` folder.
3. Name of galaxy(s) should be added to `targets.txt` (one galaxy per line).
4. Run `bash run_starcnet.sh`

**Note:** The `.tab` file must have 3 columns, first one with ids and the last two with the coordinates. If your catalog only has the two columns of the coordinates you can use `src/add_ids_to_coords.py` file to add id column.

### Run StarcNet with online LEGUS catalogs

1. Name of galaxy(s) should be in `targets.txt`.
2. Links to the mosaic(s) `.tar.gz` files should be in `frc_fits_links.txt` (one link per line).
3. Save catalog `.tab` file of each galaxy into `legus/tab_files/` folder.
4. Run `bash run_starcnet.sh 1`

**Note:** The `.tab` file must have 3 columns, first one with ids and the last two with the coordinates. If your catalog only has the two columns of the coordinates you can use `src/add_ids_to_coords.py` file to add id column.

## Cite

If you find this code useful in your research, please consider citing:
```
@article{pmcmjas_apj2021,
	doi = {10.3847/1538-4357/abceba},
	url = {https://doi.org/10.3847/1538-4357/abceba},
	year = 2021,
	month = {feb},
	publisher = {American Astronomical Society},
	volume = {907},
	number = {2},
	pages = {100},
	author = {Gustavo P{\'{e}}rez and Matteo Messa and Daniela Calzetti and Subhransu Maji and Dooseok E. Jung and Angela Adamo and Mattia Sirressi},
	title = {{StarcNet}: Machine Learning for Star Cluster Identification},
	journal = {The Astrophysical Journal}}
```

## Acknowledgements

This work is supported by the [National Science Foundation (NSF)](https://nsf.gov/index.jsp) of the United States under the award [\#1815267](https://nsf.gov/awardsearch/showAward?AWD_ID=1815267).
