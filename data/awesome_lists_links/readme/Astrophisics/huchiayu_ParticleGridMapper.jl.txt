# ParticleGridMapper

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://huchiayu.github.io/ParticleGridMapper.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://huchiayu.github.io/ParticleGridMapper.jl/dev)
[![Build Status](https://github.com/huchiayu/ParticleGridMapper.jl/workflows/CI/badge.svg)](https://github.com/huchiayu/ParticleGridMapper.jl/actions)
[![Coverage](https://codecov.io/gh/huchiayu/ParticleGridMapper.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/huchiayu/ParticleGridMapper.jl)


```ParticleGridMapper.jl``` interpolates particle data onto either a Cartesian (uniform) grid or an adaptive mesh refinement (AMR) grid where each cell contains no more than one particle. The AMR grid can be trimmed with a user-defined maximum level of refinement. Three differnt interpolation schemes are supported: nearest grid point (NGP), smoothed-particle hydrodynamics (SPH), and Meshless finite mass (MFM). It is multi-threading parallel.

![vis_amr](https://user-images.githubusercontent.com/23061774/137218103-79a368f5-1de1-42a0-836a-3530e2a03ffa.png)

## 2:1 balance

The AMR grid can be further refined to achieve the so-called "2:1 balance" where the refinement levels of neighboring cells differ by no greater than a factor of two, which leads to a smoothed transition of refinement.

![2to1balance_medium](https://user-images.githubusercontent.com/23061774/137220920-c9c07570-d658-4fb8-b34c-2c305196c67b.gif)


# Examples

 - [examples/example_cloud.jl](https://github.com/huchiayu/ParticleGridMapper.jl/blob/master/examples/example_cloud.jl) demonstrates the usage of five different interpolation schemes: (1) Cartesian mesh with NGP; (2) Cartesian mesh with SPH; (3) NGP on an adaptive mesh; (4) SPH on an adaptive mesh; (5) MFM on an adaptive mesh. It then generates the following plot:

![compare_MFM_SPH_NGP_cloud](https://user-images.githubusercontent.com/23061774/148554036-fe67d4b4-5f02-41bf-a079-7f93c46ec2d7.png)

 - [examples/gadget2radmcAMR.jl](https://github.com/huchiayu/ParticleGridMapper.jl/blob/master/examples/gadget2radmcAMR.jl) demonstrates how to generate an adaptive mesh from a [Gadget](https://wwwmpa.mpa-garching.mpg.de/gadget4/) snapshot (format 3) and some axillary files required for radiative transfer calculations for CO rotational line emission using the [RADMC-3D](https://www.ita.uni-heidelberg.de/~dullemond/software/radmc-3d/) code. An illustration of such calculations from [Hu et al. 2022](https://arxiv.org/abs/2201.03885):

![CO_10_21_maps](https://user-images.githubusercontent.com/23061774/148557075-d33fcdc0-862a-4e6c-bcaf-7e545c4f1ce1.png)

# Author
Chia-Yu Hu @ Max Planck Institute for Extraterrestrial Physics 
(cyhu.astro@gmail.com)
