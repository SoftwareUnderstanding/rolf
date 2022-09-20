# Paramo

"Vine a Comala porque me dijeron que acá vivía mi padre, un tal Pedro Páramo."

## What it does

Paramo stands for: PArticles and RAdiation MOnitor. In a few words: this code solves que Fokker-Planck equation and calculates synchrotron and inverse Compton emission.

## Prerequisites

- Fortran (either GNU or Intel)
- HDF5

## How to run it

### Customize `Makefile`

The `Makefile` has a sereies of variables that allow to compile to our convenience.

- `COMPILER`: 0 for GCC and 1 for Intel compilers
- `DEBUGGING`: Compile in debugging or optimized mode
- `USEHDF5`: Compile with HDF5 libraries and save data in that format. **NOTE**: Not all problems are 

### Compile and run

- Clone the repository
- Create parameters file
- `make`
- Run executable

## Example

A fast way to run `Paramo` can be done by downloading/copy the file [`runGRB190114C.py`](https://bitbucket.org/comala/workspace/snippets/LpLgGL/afterglow-of-grb190114c#file-runGRB190114C.py) from the [snippets](https://bitbucket.org/comala/workspace/snippets/). Running this file will:

- Create the parameters file
- Compile `Paramo` for an afterglow simulation
- Run the executable

**Note**: The code can be run from another directory. This can be done by specifying in `runGRB190114C.py` its full address, as well `Paramo`'s.

# References

These are the most referenced works on which I based all the modeling of afterglows

- [PVP14] Pannanen, Vurm, Poutanen, 2014, A&A, 564, A77
- [PM09]  Petropoulou, Mastichiadis, 2009, A&A, 507, 599
- [RM92]  Rees & Meszaros, 1992, MNRAS, 258, 41P
- [DM09]  Dermer & Menon, 2009, "High Energy Radiation from Black Holes: Gamma Rays, Cosmic Rays, and Neutrinos", Princeton Series in Astrophysics
- [SPN98] Sari, Piran, Narayan, 1998, ApJ, 497, L17
- [PK00]  Panaitescu & Kumar, 2000, ApJ, 543, 66
- [CL00]  Chavalier & Li, 2000, ApJ, 536, 195
- [CD99]  Chiang & Dermer, 1999, ApJ, 512, 699
