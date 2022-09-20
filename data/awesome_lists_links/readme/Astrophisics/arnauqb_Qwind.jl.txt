# Qwind

![CI](https://github.com/arnauqb/Qwind.jl/workflows/CI/badge.svg)
[![codecov](https://codecov.io/gh/arnauqb/Qwind.jl/branch/main/graph/badge.svg?token=KQPtxMDMAm)](https://codecov.io/gh/arnauqb/Qwind.jl)

# Qwind: Modelling UV line-driven winds in the context of Active Galactic Nuclei (AGN)


Qwind is a code that aims to simulate the launching and acceleration phase of line-driven winds in the context of AGN accretion discs. To do that, we model the wind as a set of streamlines originating on the surface of the AGN accretion disc, and we evolve them following their equation of motion, given by the balance between radiative and gravitational force.

# Setup

The code is written in the [Julia][https://julialang.org/]. If you are not familiar with it, its syntax is fairly similar to Python, but with much greater performance, so I encourage you to give it a go. Refer to the Julia webpage to installation instructions.

Qwind is not released in the official Julia package list yet, but you can quickly set it up by doing

```
git clone https://github.com/arnauqb/Qwind.jl
cd Qwind.jl
```

Then fire up Julia:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

This will automatically set you up and install all the required dependencies.

## Running the code

You first need to setup a parameter file. An example file can be found in ``configs/config_example.yaml`` which contains:

```yaml
 black_hole:
  M: 1e8 # solar masses
  mdot: 0.5 # Eddington units
  spin: 0.0 # spin parameter (-1,1)

radiation:
  relativistic: true 
  f_uv: auto
  f_x: 0.15
  n_r: 1000 # number of bins in the disk
  disk_r_in: 6.0
  z_xray: 0.0
  disk_height: 0.0
  xray_opacity: boost 
  tau_uv_calculation: disk 
  disk_integral_rtol: 1e-3
  wind_interpolator:
    update_grid_method: "average"
    vacuum_density: 1e2
    n_integrator_interpolation: 10000
    nz: 500
    nr: auto 

grid:
  # units of Rg
  r_min: 6.0
  r_max: 50000.0
  z_min: 0.0
  z_max: 50000.0

initial_conditions:
  mode: CAKIC
  r_in: 20.0
  r_fi: 1500.0
  n_lines: auto
  log_spaced: true
  z0: 0.0 # Rg
  K: 0.03
  alpha: 0.6
  mu: 0.61
  use_precalculated: true

integrator:
  n_iterations: 2
  atol: 1e-8
  rtol: 1e-3
  save_path: "./tests"
```



To run a model, it is as simple as:

```julia
using Qwind

model = Model("configs/config_example.yaml")

run!(model)
```

The results are stored in HDF5 format, in this case in the folder ```tests/results.hdf5``` . Qwind provides a nice interface to read the results as well

```julia
density_grid = DensityGrid("tests/results.hdf5")
density_grid["r_range"] # this is the radial range of the grid
density_grid["z_range"] # this is the z range of the grid
density_grid["density"] # 2D matrix with the values.

trajectories = load_trajectories("tests/results.hdf5")
# dictionary with the trajectories
```

