# _aztekas_-code 

2017 Alejandro Aguayo-Ortiz \& Sergio Mendoza and &copy; Alejandro Aguayo-Ortiz 2019-2021.

_aztekas_ is a program that solves hyperbolic partial differential equations in conservative form using High Resolution Shock-Capturing (HRSC) schemes. This version of _aztekas_ allows to solve the non-relativistic and relativistic hydrodynamic equations of motion (Euler equations) for a perfect fluid. The relativistic part can solve these equations on a background fixed metric (Schwarzschild, Minkowski, Kerr-Schild, etc.).

## Introduction

_aztekas_ uses a conservative finite-volume approach to obtain the dicrete form of a hyperbolic partial differential system of equations (PDE).

![equation](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20Q%7D%7B%5Cpartial%20t%7D%20&plus;%20%5Cfrac%7B%5Cpartial%20F%5Ei%7D%7B%5Cpartial%20x%5Ei%7D%20%3D%20S)

where ![equation](https://latex.codecogs.com/gif.latex?Q) is the vector of conservative variables (energy, density, momentum), ![equation](https://latex.codecogs.com/gif.latex?F%5Ei) are the vectors of fluxes along each dimensional direction and ![equation](https://latex.codecogs.com/gif.latex?S) is the vector of source terms.

## How to install and run an example simulation

1. Open terminal in your computer.
2. Clone github repository: `$ git clone https://github.com/aztekas-code/aztekas-main`
3. Export the environment variable AZTEKAS_PATH `$ echo 'export AZTEKAS_PATH=$HOME/aztekas-main' >> ~/.bashrc` for Linux or `$ echo 'export AZTEKAS_PATH=$HOME/aztekas-main' >> ~/.bash_profile` for Mac
4. Run `$ source ~/.bashrc`or `$ source ~/.bash_profile` in order to update the environment variable.
5. Move to SIMULATIONS/HD/Shock-Tube: `$ cd SIMULATIONS/HD/Shock-Tube`
6. Compile: `$ make clean` and `$ make`
7. Run _aztekas_: `./aztekas shock.param`

## Setup a new simulation

1. Run the command `$ chmod 755 setup` inside the main directory.
2. Create a new directory wherever you want.
3. Run the `setup` script inside this new repository and setup the files for the new simulation: `$ $AZTEKAS_PATH/setup`
