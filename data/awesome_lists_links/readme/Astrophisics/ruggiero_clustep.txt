## Description

Script that generates a snapshot in the GADGET-2 format containing a galaxy cluster
halo in equilibrium. The halo is made of a dark matter component and a gas component,
with the latter representing the ICM. Each of these components follows a Dehnen
density profile ([Dehnen 1993](http://adsabs.harvard.edu/abs/1993MNRAS.265..250D)),
with gamma=0 or gamma=1. If gamma=1, then the profile corresponds to a Hernquist
profile ([Hernquist 1990](http://adsabs.harvard.edu/abs/1990ApJ...356..359H)).
See [Ruggiero & Lima Neto (2017)](http://adsabs.harvard.edu/cgi-bin/bib_query?arXiv:1703.08550)
for a discussion on the difference between these two options.

The value for the gravitational constant G used in this code is such that
the unit for length is 1.0 kpc, for mass 1.0e10 solar masses, and for
velocity 1.0 km/s. This is the default for GADGET-2, and works out of the
box in RAMSES with the [DICE patch](https://bitbucket.org/vperret/dice/wiki/RAMSES%20simulation).


## Required libraries
(and the names of the packages in Debian-like systems)
 
* NumPy (python-numpy)
* SciPy (python-scipy)
* Cython (cython), higher than 0.17.4
* Matplotlib (python-matplotlib)
* Argparse (python-argparse), in case you use Python < 3.2
* python-dev


## Installation

This code doesn't need to be installed, but a custom Cython
library which is included has to be compiled. For that, just cd to
`/cluster` and run `make`. A new file, named `optimized_funcions.so`,
will be created, and then `clustep.py` will be ready for execution.


## Usage

You can run `python clustep.py --help` to see the message below. 
Please check out the `params_cluster.ini` file to see the available free parameters.

    usage: clustep.py [-h] [--no-dm] [--no-gas] [-o init.dat]
    
    Generates an initial conditions file for a galaxy cluster halo simulation.
    
    optional arguments:
      -h, --help   show this help message and exit
      --no-dm      No dark matter particles in the initial conditions. The dark
                   matter potential is still used when calculating the gas
                   temperatures.
      --no-gas     Gas is completely ignored, and only dark matter is included.
      -o init.dat  The name of the output file.

Some analysis scripts are also included in the `analysis/` folder, you can try
these out. I haven't documented them because they are changed all the time and
aren't all that well written as of now.


## Works which used this code

* [Ruggiero & Lima Neto (2017)](http://adsabs.harvard.edu/cgi-bin/bib_query?arXiv:1703.08550)


## Acknowledgements

Credits for Prof. Dr. Rubens Machado (http://paginapessoal.utfpr.edu.br/rubensmachado),
for the vital support and suggestions in writing this code.

## Disclaimer

Feel free to use this code in your work, but please link this page
in your paper.
