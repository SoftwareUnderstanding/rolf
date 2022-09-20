# Large-Scale Structure Information

This Fisher code produces forecasts for the LSST 3x2 point functions analysis, or the LSSTxCMB S4 and LSSTxSO 6x2 point functions analyses.
It computes the auto and cross correlations of galaxy number density, galaxy shear and CMB lensing convergence.

Requires pathos for multiprocessing (https://pypi.org/project/pathos/): 
```
pip install pathos
```
Requires classylss for the cosmology (https://classylss.readthedocs.io/en/stable/):
```
conda install -c bccp classylss
```
or 
```
pip install classylss
```
Requires the Monte Carlo library vegas for the CMB lensing noise calculations (https://pypi.org/project/vegas):
```
pip install vegas
```


Just clone and run: 
```
python driver_fisher_lsst.py
```
You will need to switch the "save" flags to True, and uncomment any plotting comment you like to generate the plots.

Hope you find this code useful! Please cite https://arxiv.org/abs/2007.12795 if you use this code in a publication. Do not hesitate to contact me with any question: eschaan@lbl.gov
