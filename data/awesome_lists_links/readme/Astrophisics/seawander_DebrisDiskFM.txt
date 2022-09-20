# DebrisDiskFM [![DOI](https://zenodo.org/badge/141328805.svg)](https://zenodo.org/badge/latestdoi/141328805) <a href="http://ascl.net/2001.008"><img src="https://img.shields.io/badge/ascl-2001.008-blue.svg?colorB=262255" alt="ascl:2001.008" /></a>
Forward modeling for circumstellar debris disks in scattered light.

Method: Monte Carlo Markov Chain (MCMC) with one of the two disk modeling software
- the MCFOST disk modeling software, see [here](http://ipag.osug.fr/public/pintec/mcfost/docs/html/index.html) for the documentation of MCFOST.
- the Python-based disk modeling code using Henyey–Greenstein phase function by M. Millar-Blanchaer, see [here](https://github.com/maxwellmb/anadisk_model) for the code, and [here](https://ui.adsabs.harvard.edu/abs/2015ApJ...811...18M/abstract) for the original usage of the code.

DebrisDiskFM is first used in Ren, Choquet, Perrin et al. (2019) ([ADS link](https://ui.adsabs.harvard.edu/abs/2019ApJ...882...64R/abstract), [BibTeX](https://ui.adsabs.harvard.edu/abs/2019ApJ...882...64R/exportcitation)) with MCFOST, and in Ren, Choquet, Perrin et al. (2021) ([ADS link](https://ui.adsabs.harvard.edu/abs/2021ApJ...914...95R/abstract), [BibTeX](https://ui.adsabs.harvard.edu/abs/2021ApJ...914...95R/exportcitation)) with the M. Millar-Blanchaer code.

## 0. Installation
```pip install --user -e git+https://github.com/seawander/DebrisDiskFM.git#egg=Package```

The above command does not require administrator access, and can be run both on one's personal desktop and on a computer cluster.

## 1. Parameter File Setup (for MCFOST)
#### 1.1 MCFOST Parameter File Template
For a given sytem, generate a disk template with the following command
```python
sampleTemplate = mcfostParameterTemplate.generateMcfostTemplate(n_zone = n_zone, n_species = n_species, n_star = n_star)
```
The above command generate a template in the ```collections.OrderedDict()``` structure, for this template, there are ```n_zone``` zones; ```n_species``` species (note: ```n_species``` is a list. For example, when ```n_zone = 2```, we can let ```n_species = [3, 2]```, then in the 0th zone, i.e., ```zone0```, there are 3 species of grains, and in the 1st zone, i.e., ```zone1```, there are 2 species; and ```n_star``` stars shining the whole system.
#### 1.2 MCFOST Parameter File Sample for a Specific Target
From the previous paragraph, we have a sample template, then we should modify the parameters in the template for our specific target.

The structure of the template is in this mind-map-structured PDF file: [MCFOST Parameter OrderedDict.pdf](https://github.com/seawander/DebrisDiskFM/blob/master/MCFOST%20Parameter%20OrderedDict.pdf). In this PDF file, the **quoted** parameters are what you can modify, and all of the parameters have the ***same names*** as the MCFOST parameter file. The *only* added ones are the row numbers in each block (named as ```'rowW'``` where W = 0 to the number of rows - 1 in that block), and ```'zoneX'``` where X = 0 to (n_zone - 1), ```'speciesY'``` where Y = 0 to (n_species[X] - 1), and ```'starZ'``` where Z = 0 to (n_star - 1).

For example, if you want to turn on the Stokes maps, use 
```python
sampleTemplate['#Wavelength']['row3']['stokes parameters?'] = 'T'
```

or if you want to change the input file for optical indices to 'ice_opct.dat' for the 2nd species in zone0, use

```python
sampleTemplate['#Grain properties']['zone0']['species2']['row1']['Optical indices file'] = 'ice_opct.dat'
```

for the detailed parameters that can be changed, refer to the PDF file.

#### 1.3 Save parameter file
Just call 
```python
save_path = None
mcfostParameterTemplate.display_file(sampleTemplate, save_path)
```
and it will save the parameter structure in a proper MCFOST parameter file format to ```save_path```, if ```None``` then it will display to the screen only; or save to the address if it is not ```None```.

## 2. Markov chain Monte Carlo (MCMC) [debris disk](https://en.wikipedia.org/wiki/Debris_disk) radiative transfer modeling framework set-up
### 2.1 Basic Baysian Statistics Knowledge
From the conditional probability equation, 
<p align="center">P(A|B) = P(AB)/P(B) = P(B|A)P(A)/P(B), </p>
we have 
<p align="center">P(B|A) = P(A|B)P(B)/P(A). </p>
Let A be the observed data, and B be the hidden parameters, then we can infer the distribution of B from the A data we have from the above equation. However, in most cases, we do not know the distribution of A, the above equation can be written as
<p align="center">P(B|A) ∝ P(A|B) P(B),</p>

which is the famous [posterior probability](https://en.wikipedia.org/wiki/Posterior_probability) relationship, i.e.,
<p align="center">Posterior probability ∝ Likelihood x Prior probability. </p>

We can use [MCMC](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo) for to extract the posterior probability distribution for the unknown parameters. In reality, we usually take a log function on both sides for the above equation, which then transforms multiplication to addition, i.e.,

<p align="center"> log(Posterior probability) = log(Likelihood) + log(Prior probability) + constant.</p>


### 2.2 Setting up the MCMC Framework for Debris Disk Modeling
Now we use the HD191089 debris disk system as an example to explain the DebrisDiskFM framework.
#### 2.2.1 Prior Distribution

Define the variable names and prior values as in [debrisdiskfm/lnprior.py](https://github.com/seawander/DebrisDiskFM/blob/master/debrisdiskfm/lnprior.py). In the ```lnprior_hd191089``` function, the default prior distribution is uniform distribution. 

Please change or adjust the names/values/distribution for your own system. For a uniform prior, if a value is drawn outside the range, then negative infinity is returned. 

Note: the variable names should match the names in the ```#### Section 2: Variables ####``` section in the [debrisdiskfm/mcfostRun.py](https://github.com/seawander/DebrisDiskFM/blob/master/debrisdiskfm/mcfostRun.py) ```run_hd191089``` function. As long as the names are matching between the two files, you can assign the names as what you want.

#### 2.2.2 Log-likelihood

The log-likelihood script is in [debrisdiskfm/lnlike.py](https://github.com/seawander/DebrisDiskFM/blob/master/debrisdiskfm/lnlike.py), where the ```lnlike_hd191089``` function defines the log-likelihood of your system. The ```lnlike_hd191089``` function has two major sections: the ```### Observations``` section, and the ```### (Forwarded) Models``` section.

In the ```### Observations``` section, the observed data and the corresponding uncertainties are loaded.

In the ```### (Forwarded) Models``` section, the simulated data are loaded (and convolved with a point source point-spread-function) to simulate the observation.

The observed data and models are then combined to calculate log-likelihood. Note: the ```chi2``` function in [debrisdiskfm/lnlike.py](https://github.com/seawander/DebrisDiskFM/blob/master/debrisdiskfm/lnlike.py) returns the log-likelihood by default; and more importantly, for [normal distribution](https://en.wikipedia.org/wiki/Normal_distribution), which is our assumption of how the unknown parameters are distributed, the [chi-squared value](https://en.wikipedia.org/wiki/Chi-squared_test) is the log-likelihood added with some constants.

#### 2.2.3 Poster Distribution

The script for calculating the log posterior distribution is in [debrisdiskfm/lnpost.py](https://github.com/seawander/DebrisDiskFM/blob/master/debrisdiskfm/lnpost.py), and the ```lnpost_hd191089``` function is used for the HD191089 system.

The ```lnpost_hd191089``` function takes the input variable names and values, calculate the [MCFOST](http://ipag.obs.ujf-grenoble.fr/~pintec/mcfost/docs/html/overview.html) models using ```mcfostRun.run_hd191089```, then calculate the log-likelihood, then add the prior with the log-likelihood to obtain the posterior.

### 2.3 Running MCFOST for the MCMC Framework
As mentioned in 2.2.3, the ```mcfostRun.run_hd191089``` function is where the MCFOST models are generated. The script is in [debrisdiskfm/mcfostRun.py](https://github.com/seawander/DebrisDiskFM/blob/master/debrisdiskfm/mcfostRun.py), however, it can be generalized to use any radiative transfer modeling software, as long as the input parameter files are also correctly modified.

In the ```run_hd191089``` function of [debrisdiskfm/mcfostRun.py](https://github.com/seawander/DebrisDiskFM/blob/master/debrisdiskfm/mcfostRun.py), there are 4 major sections: ```Section 1: Fixed Parameters```, ```Section 2: Variables ```, ```Section 3: Parameter File for HD191089```, and ```Section 4: Run```.

Section 1 sets the fixed parameters for the system;

Section 2 takes the input ```var_names``` and ```var_values``` the modify the corresponding parameters;

Section 3 save the MCFOST parameter files for different instruments to ```paraPath``` (the path can be hashed for a cluster, and this will prevent multiple MCFOST runs accessing the same folder, this is by default set to ```True```);

Section 4 run the MCFOST radiative transfer modeling using the parameter files generated from Section 1 to 3.

## 3. Run MCMC
The are two exampels in [debrisdiskfm](https://github.com/seawander/DebrisDiskFM/tree/master/debrisdiskfm): [main_laptop.py](https://github.com/seawander/DebrisDiskFM/blob/master/debrisdiskfm/main_laptop.py) for running the codes on a laptop/desktop which does not involve multiple nodes but handles multiple cores on a single node, and [main_cluster.py](https://github.com/seawander/DebrisDiskFM/blob/master/debrisdiskfm/main_cluster.py) which handles the case for multiple nodes on a computer cluster.

In [main_cluster.py](https://github.com/seawander/DebrisDiskFM/blob/master/debrisdiskfm/main_cluster.py), if you are limited by the run time for a single job, this script is able to handle that by saving the progress then load it again in the next run. This is enabled using the backend function in the 3.0rc1 version of [emcee](http://dfm.io/emcee/current/) with [h5py](http://www.h5py.org) installed. **To enable this, please install the latest 3.0rc1 version of the [emcee](http://emcee.readthedocs.io/en/latest/user/install/) software** by 
```python
git clone https://github.com/dfm/emcee.git
cd emcee
python setup.py install
```

The default status is stored in ```filename```, you can change the variable in that line in [main_cluster.py](https://github.com/seawander/DebrisDiskFM/blob/master/debrisdiskfm/main_cluster.py) per your own discretion.  If the file is not deleted, then for a new run, emcee will automatically load it and start calculation from there.

The other parameters to change are: ```n_walkers``` which defines the number of walkers (10 times of the dimension or more is suggested), and ```step``` which denotes how many MCMC steps do you want to perform in *this* run.

To extract the information from the backend file, such as the corner plot, please refer to the emcee [webpage](https://emcee.readthedocs.io/en/latest/tutorials/monitor/) for details.

***Example: Deploy the code to a Slurm cluster with mpi4py***:
Create a file named ```sub_mcmc_cluster```, with the contents: 
```bash
#!/bin/bash -l

#SBATCH --partition=parallel
#SBATCH --job-name=mcmc_cluster
#SBATCH --time=0:10:0
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=6       
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G

export MKL_NUM_THREADS=4
mpiexec -n 12 python3 -W ignore main_cluster.py
```
which used 2 nodes, and 6 tasks on each node, with each task using 4 cores. Now submit it in the command line using

```bash
sbatch sub_mcmc_cluster
```
and wait for the results!

For more detailed explanation of the script, please go to [DebrisDiskFM/cluster_example_emcee/](https://github.com/seawander/DebrisDiskFM/tree/master/cluster_example_emcee).

*BibTex*:
```
@misc{debrisdiskFM,
  author       = { {Ren}, Bin and {Perrin}, Marshall },
  title        = {DebrisDiskFM, v1.0, Zenodo,
doi: \href{https://zenodo.org/badge/latestdoi/141328805}{10.5281/zenodo.2398963}. },
  version = {1.0},
  publisher = {Zenodo},
  month        = Dec,
  year         = 2018,
  doi          = {10.5281/zenodo.2398963},
  url          = {https://doi.org/10.5281/zenodo.2398963}
}
```
