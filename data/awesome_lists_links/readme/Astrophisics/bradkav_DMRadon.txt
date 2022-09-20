# DMRadon

[![arXiv](https://img.shields.io/badge/arXiv-1502.04224-B31B1B.svg)](https://arxiv.org/abs/1502.04224) [![ASCL](https://img.shields.io/badge/ascl-2002.012-blue.svg?colorB=262255)](https://ascl.net/2002.012) [![MIT Licence](https://badges.frapsoft.com/os/mit/mit.svg?v=103)](https://opensource.org/licenses/mit-license.php)

Tools for calculating the Radon Transform, for use in the analysis of Directional Dark Matter Direct Detection. Currently the code allows you to calculate speed distributions and Radon Transforms only for a standard Maxwell-Boltzmann distribution.

**To be added soon:** 
- Calculation of generic Radon Transform, especially for discretised velocity distributions (see [arXiv:1502.04224](https://arxiv.org/abs/1502.04224)).
- Notes and references for all calculations.

## Getting started

The code is in python and you'll need some standard python libraries: `numpy` and `scipy`.

The routines are in the `MaxwellBoltzmann.py` module (for now). So just import that the module:
```python
import MaxwellBoltzmann as MB
```
and you're ready to go. For example usage, check out the jupyter notebook, [Examples-MB.ipynb](/Examples-MB.ipynb), which can also be viewed, in browser, [here](https://nbviewer.jupyter.org/github/bradkav/DMRadon/blob/master/Examples-MB.ipynb).

## Functionality

So far, you can use the code to calculate the Maxwell-Boltzmann velocity distribution, speed distribution, velocity integral (eta) and ***Radon Transform***. You can also calculate the velocity distribution *averaged over different angular bins*, which may prove useful if you want to look at calculating binned/integrated Radon Transforms (which will be added soon).

The procedures are:

```python
VelDist(v, theta, phi, vlag=220.0, sigv=156.0, vesc=533.0)
SpeedDist(v, vlag=220.0, sigv=156.0, vesc=533.0)
VelDist_avg(v, j, N_bins,vlag=220.0, sigv=156.0, vesc=533.0)
Eta(vmin, vlag=220.0, sigv=156.0, vesc=533.0)
RadonTransform(vmin, theta, phi, vlag=220.0, sigv=156.0, vesc=533.0)
```

The arguments should be reasonably self explanatory (and I'll add better documentation soon). In the case of `VelDist_avg`, the function returns the velocity distribution, averaged over an angular bin. In this case, the directions are divided into `N_bins` bins and the average is taken over bin number `j = 1, 2,.., N_bins`, where `j = 1` is centred on the median DM velocity and `j = N_bins` is centred on the opposite direction.

Most of the functions accept the following option arguments:
- `vlag = 220.0`, the speed of the lab with respect to the DM halo (the 'lag speed')  
- `sigv = 156.0`, the standard deviation of the Maxwell-Boltzmann  
- `vesc = 533.0`, the escape speed in the Galactic frame

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
