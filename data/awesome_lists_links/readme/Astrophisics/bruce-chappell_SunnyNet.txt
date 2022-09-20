# SunnyNet


A neural network framework for solving 3D NLTE radiative transfer in stellar atmospheres.

Uses Python 3.8.5, Pytorch 1.6.0, Helita 0.9.0

[![DOI](https://zenodo.org/badge/367666957.svg)](https://zenodo.org/badge/latestdoi/367666957)

## Usage

SunnyNet is used to learn the mapping the between LTE and NLTE populations of a model atom, and predict the NLTE populations based on LTE populations for an arbitrary 3D atmosphere. To use SunnyNet, one must already have a set of LTE and NLTE populations computed in 3D, to train the network. These must come from another code, as SunnyNet is unable to solve the formal problem. Once SunnyNet is trained, one can feed it LTE populations from a different 3D atmosphere, and obtain predicted NLTE populations. The NLTE populations can then be used to synthesize any spectral line that is included in the model atom. SunnyNet does not read the model atom; its details are unimportant for the task.

The output of SunnyNet is a file with predicted NLTE populations. SunnyNet itself does not calculate synthetic spectra, but we include a sample script written in the Julia language to quickly compute Hα spectra, making use of [Transparency.jl](https://github.com/tiagopereira/Transparency.jl).

Using SunnyNet can be divided in the following steps, using functions from `SunnyNet.py`:

1. Prepare a file for training using one or more existing 3D atmospheres with known LTE and NLTE populations. From the atmospheres one needs the height scale (in m), and the horizontally-averaged mass density as function of height (in kg m^-3). This is needed to interpolate the populations to a common grid of mean column mass. For this step, use `build_training_set()`. 

2. Using the previously-prepared file, train SunnyNet for a given network model. For example, 'SunnyNet_3x3' uses a 3x3 window around the column of interest, 400 depth points and 6 atomic levels. For this step, use `sunnynet_train_model()`.

3. Having trained SunnyNet, the training files can be used to learn the conversion from LTE to NLTE populations in 3D. Using a different atmosphere than the one used for training, first prepare a file that will be used as input for the network. You will need the LTE populations and, as before, the height scale and horizontally-averaged mass density from a 3D atmosphere. For this step, use `build_solving_set()`.

4. Once you have the prepared file in SunnyNet format, the next step is to predict the NLTE populations. Using the files from the previous two steps, use `sunnynet_predict_populations()`. 

The output file contains the NLTE populations for all columns, but the height dimension is on the network's mean column mass scale, not geometrical height as in the input 3D atmosphere. If needed, it can be interpolated back to the original geometrical height scale using the saved column mass scales of SunnyNet and the 3D model (see `read_sunnynet_pops()` in `calc_intensity.jl`, which does this)


### Example with MULTI3D output

We include a sample example script for running SunnyNet with output from MULTI3D. This makes use of the [helita](https://github.com/ita-solar/helita) package to read the output of MULTI3D.

```python
import SunnyNet
import numpy as np
from helita.sim.multi3d import Multi3dAtmos, Multi3dOut

# 1. Prepare training data
multi3d_path = "/path/to/3D_sim/snapshot_%i/output/"
multi3d_atmos = '/path/to/3D_sim/atm3d.3D_sim_snapshot%i'
training_snapshots = [1, 2, 3]
lte_pops = []
nlte_pops = []
rho_mean = []
z_scale = []
for s in training_snapshots:
    m3d = Multi3dOut(directory=multi3d_path % s)
    m3d.readall()
    lte_pops.append(m3d.atom.nstar * 1e6)  # cm^-3 to m^-3
    nlte_pops.append(m3d.atom.n * 1e6)
    nx, ny, nz, nlevel = m3d.atom.nstar.shape
    atmos = Multi3dAtmos(multi3d_atmos % s, nx, ny, nz)
    rho = atmos.rho
    rho_mean.append(np.mean(rho, axis=(0,1)) * 1e3)  #  g cm-3 to kg m-3
    z_scale.append(m3d.geometry.z * 1e-2)  # cm to m

SunnyNet.build_training_set(lte_pops, nlte_pops, rho_mean, z_scale, 
              save_path='3D_sim_train_s123.hdf5', ndep=400, pad=1, tr_percent=85)

# 2. Train network
SunnyNet.sunnynet_train_model('3D_sim_train_s123.hdf5', 'training/', 
                              '3D_sim_train_s123.pt', model_type='SunnyNet_3x3',alpha=0.2, cuda=True)


# 3. Prepare data for predicting
pred_snap = 5
m3d = Multi3dOut(directory=multi3d_path % pred_snap)
m3d.readall()
lte_pops = m3d.atom.nstar * 1e6
nx, ny, nz, nlevel = lte_pops.shape
atmos = Multi3dAtmos(multi3d_atmos % pred_snap, nx, ny, nz)
rho = atmos.rho
rho_mean = np.mean(rho, axis=(0,1)) * 1e3
z_scale = m3d.geometry.z * 1e-2
SunnyNet.build_solving_set(lte_pops, rho_mean, z_scale, 
                           save_path='3D_sim_predict_s%i.hdf5' % pred_snap, ndep=400, pad=1)


# 4. Predict populations
SunnyNet.sunnynet_predict_populations(
    "training/3D_sim_train_s123.pt",
    "3D_sim_train_s123.hdf5",
    "3D_sim_predict_s5.hdf5",
    "sunnynet_output_3D_sim_s5.hdf5",
)
```