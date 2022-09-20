Zwindstroom
===========

The purpose of Zwindstroom is to provide methods for computing background
quantities and scale-dependent growth factors for massive neutrino cosmologies.
Following the earlier REPS code (Zennaro et al., 2016), we use a Newtonian
fluid approximation with an external neutrino sound speed to close the Boltzmann
hierarchy (Shoji & Komatsu 2010; Blas et al. 2014). Zwindstroom supports
multi-fluid models with distinct transfer functions and sound speeds. A flexible
python interface makes it easy to interact with CLASS (through classy). There is
also a Zwindstroom plugin for the cosmological initial conditions generator
monofonIC that allows for higher-order LPT ICs for massive neutrino
simulations in a single step.

Quick Installation
------------------

First build the C library with

```
mkdir build && cd build
cmake ..
make
cd ..
```

Then the python package can be loaded with

```python
import zwindstroom
```

Requirements
------------
+ GSL
+ CLASS / classy (optional)

Example
-------

The following snippet computes the present-day mass fraction of
non-relativistic neutrinos for a given cosmological model.

```python
from zwindstroom import *

# The neutrino species
M_nu = [0.05, 0.07] # eV
deg_nu = [2.0, 1.0] # degeneracies
N_nu = len(M_nu)

# Initialise a unit system (default uses Mpc lengths and km/s velocities)
unit_system, physical_consts = units.init_units()

# We want to integrate the cosmological tables starting at this scale factor
a_start = 1e-3

# Set up a cosmological model
params = {"h": 0.67,
          "Omega_b": 0.048,
          "Omega_c": 0.242,
          "N_nu": N_nu,
          "M_nu": M_nu,
          "deg_nu": deg_nu,
          "T_nu_0": 1.95,
          "T_CMB_0": 2.728,
          "w0": -1.0}
model = cosmology.MODEL()
model.set(params)
model.compute(unit_system, physical_consts, a_start)

f_nu = model.get_f_nu_nr_tot_of_a(1.0)
print("The neutrino fraction is:", f_nu)
```

See `example_growth_factors_class.py` for a calculation of growth factors using
growth rates from classy.
