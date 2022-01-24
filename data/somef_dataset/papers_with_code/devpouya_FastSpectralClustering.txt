# ASL project: Spectral Clustering

# Table of Contents

1. [Introduction](#Introduction)
3. [Installation](#Installation)
4. [Commands](#Commands)

<a name="Introduction"></a>
## Introduction


This project implements the Spectral Clustering algorithm presented in the following paper: https://arxiv.org/abs/0711.0189.
Refer to 18.pdf for detailed information and analysis.

![6 clusters](output/validation/6_c.png) 





<a name="Installation"></a>
## Installation

On linux, run: `./requirements/install.sh`

#### Python requirements

`pip install -r requirements/python-requirements.txt`

#### C requirements

##### linux
`sudo apt-get install libeigen3-dev`

The following are optional and only needed if you plan on using ARPACK or LAPACK

Optional: `git clone https://github.com/xianyi/OpenBLAS.git`, `cd openblas`, `make`

Optional: `sudo apt-get install libopenblas-dev`

Optional: `sudo apt-get install $(cat requirements/apt-requirements.txt)`

Optional: `sudo apt-get install liblapacke-dev`

##### macOS

`brew install gcc-9 eigen`

Optionally also follow the instructions in README-ARPACK if you plan on using ARPACK.

If you plan on using LAPACK

`brew install openblas`


##### Once requirements are installed

`make init-spectra`


<a name="Commands"></a>
## Compile


Compile with `make`


##### Run 

- `./clustering <path/Dataset.txt> <# of clusters> <path/Output.txt>`
- e.g.: 
- ` ./clustering ./datasets/test_points/6_c.txt 6 out.txt
`

##### Visualization

- interactive shell: `python3 scripts/validation/validation.py`
    - just specify names, input path is **/datasets/test_points/**, output is **/output/validation/**

**To generate data sets (interactive):**

- `generate_gaussian.py`

