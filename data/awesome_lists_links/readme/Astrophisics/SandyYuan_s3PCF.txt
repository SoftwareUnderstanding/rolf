# s3PCF: a package for computing the 3PCF in the squeezed limit

## Authors:
Sihan (Sandy) Yuan, Daniel Eisenstein & Lehman Garrison

## Introduction:
A code that computes the 3-point correlation function (3PCF) in the squeezed limit given galaxy positions and pair positions. This statistics is described in detail in [Yuan, Eisenstein & Garrison (2017)](https://arxiv.org/abs/1705.03464). 

The code is currently written specifically for the [Abacus simulations](https://lgarrison.github.io/AbacusCosmos/), but the main functionalities can be also easily adapted for other galaxy catalogs with the appropriate properties. 

## Usage:

The code does not currently have dependencies other than basic Python packages. 

To install, simply download the .py files to the directory you want the mock catalogs to live in. If you are not on the Eisenstein Group computer clusters at CfA, you may need to change the `product_dir` variable to point to the location of the simulation data. 

If you wish to access the package anywhere on your system, simply add `export PYTHONPATH="/path/to/s3PCF:$PYTHONPATH"` to your .bashrc file. Modify `/path/to/` to the directory of the package.

### Input:
The main interface of the code is the function `calc_qeff()`, which takes the following inputs:
- `whichsim` : integer. The index of the simulation box. For the Abacus 1100/h Mpc simulation with Planck cosmology, this number ranges from 0 to 15.  
- `pos_full` : dictionary. The baseline HOD parameters. The dictionary requires the following five parameters:
- `pair_data` : numpy.array. Array of pair data containing the following seven columns: x (Mpc), y (Mpc), z (Mpc), dist (Mpc), id1, id2, mhalo (Msun). The ids are the galaxy id number of the two galaxies in the pair. Do not include duplicate pairs in the pair data. 
- `params` : dictionary. Simulation parameters. Following are the required parameters:
  - `z` : float. Redshift of the simulation snapshot. With the current directory, `z = 0.5`.
  - `h` : float. Hubble constant. For Planck cosmology, we use `h = 0.6726`.
  - `Lbox` : float. The size of the simulation box in Mpc. For current Abacus 1100 boxes, `Lbox = 1100/h`.
  - `Mpart` : float. The particle mass in solar mass. For current Abacus 1100 boxes, `Mpart = 3.88e10/h`.
  - `velz2kms` : float. The conversion from simulation velocity to velocity in km/s and vice versa. The value can be calculated by H(z)/(1+z). 
  - `num_sims` : integer. The number of simulation boxes. For current Abacus 1100 boxes, `num_sims = 16`.
- `dist_nbins` : integer. The number of bins in pair separation along the parallel and perpendicular direction.
- `whatseed` : integer (optional). The seed to the random number generator. Default value is 0.
- `rsd` : boolean (optional). The redshift space distortion flag. Shifts the LOS locations of galaxies. Default is True. 
```python
import numpy as np
import os,sys

from s3PCF import calc_pair_bias as calcbias


# constants
params = {}
params['z'] = 0.5
params['h'] = 0.6726
params['Nslab'] = 3
params['Lbox'] = 1100/params['h'] # Mpc, box size
params['Mpart'] = 3.88537e+10/params['h'] # Msun, mass of each particle
params['velz2kms'] = 9.690310687246482e+04/params['Lbox'] # H(z)/(1+Z), km/s/Mpc
params['maxdist'] = 30 # Mpc # use 10 Mpc for the real space case
params['num_sims'] = 16

# the standard HOD, Zheng+2009, Kwan+2015
M_cut = 10**13.35 # these constants are taken at the middle of the design, Kwan+15
log_Mcut = np.log10(M_cut)
M1 = 10**13.8
log_M1 = np.log10(M1)
sigma = 0.85
alpha = 1
kappa = 1
A = 0

# rsd?
rsd = True
params['rsd'] = rsd

whichsim = 0

# the data directory 
datadir = "../gal_profile/data"
if rsd:
    datadir = datadir+"_rsd"
savedir = datadir+"/rockstar_"+str(M_cut)[0:4]+"_"+str(M1)[0:4]+"_"+str(sigma)[0:4]+"_"+str(alpha)[0:4]+"_"+str(kappa)[0:4]+"_"+str(A)
if rsd:
    savedir = savedir+"_rsd"

# load the galaxy and pair data
print "Loading pair/galaxy catalogs...", whichsim
pos_full = np.fromfile(savedir+"/halos_gal_full_pos_"+str(whichsim))
pair_data = np.fromfile(savedir+"/halos_pairs_full_"+str(whichsim)+"_maxdist"+str(int(params['maxdist'])))
pos_full = np.array(np.reshape(pos_full, (-1, 3))) / params['Lbox'] - 0.5 # relative unit
pair_data = np.array(np.reshape(pair_data, (-1, 7)))

# compute pair bias and Qeff
calcbias.calc_qeff(whichsim, pos_full, pair_data, params, rsd = params['rsd'])
```

### Output:
There is one main output text file. It contains the pair-galaxy bias and squeezed 3PCF info in all the bins. 
Specifically it contains the following columns. 
- ilos   : the bin index along the line of sight.
- itrans : the bin index perpendicular to the line of sight. 
- bpg    : the pair-galaxy bias for the bin. 
- Qeff   : the squeezed 3PCF for the bin.
- npairs : the number of pairs in the bin. 


## Citation:
If you use this code, please cite [Yuan, Eisenstein & Garrison (2017)](https://arxiv.org/abs/1705.03464).

## Help 
If you need help with the code, please contact me (Sandy) at sihan.yuan@cfa.harvard.edu. 
