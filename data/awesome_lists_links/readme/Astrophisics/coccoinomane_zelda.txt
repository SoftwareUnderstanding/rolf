## DESCRIPTION

Zelda is a command-line tool to extract correlation functions in velocity
space from a galaxy catalogue. Being modular and extendible, Zelda can
be generalized to produce power spectra and to work in position
space as well.

The original contributors (Guido W. Pettinari, Alicia Bueno Belloso and
Nikolai Meures) wrote Zelda to analyze the infall velocity of isolated
galaxy pairs in the Millennium simulation, a project in collaboration
with Will Percival that led to the publication of a paper 
(http://arxiv.org/abs/1204.5761).

Zelda is written in C. Its structure is modular and flexible, and it was heavily
inspired by that of the cosmological Boltzmann code CLASS (http://class-code.net/).

Zelda is a parallel code, via the OpenMP standard. Therefore, it can use all the
cores in a single node, but it cannot span multiple nodes. Set the number of
cores you want to use on the command line via the environment variable
`OMP_NUM_THREADS`. For example, if you run Zelda on a laptop with 8 cores
on a bash shell, you might want to execute `export OMP_NUM_THREADS=8` before
running Zelda.


## CONTRIBUTE!

Zelda is now open source. The whole project is hosted on a public repository
on Github, at the following link:

https://github.com/coccoinomane/zelda

Feel free to download the code, test it, and modify it! If you want to
share your modifications, we are happy to make you a collaborator of
the project, or to accept pull requests. 


## INSTALLATION

Zelda can be installed using GNU make. Personalise the 'makefile' if
you want to use a C compiler different from the default GCC. To compile,
just run 'make all'.

If that does not work, try first running `make clean` and then again
`make all`. If that does not work either, make sure that you have installed
the OPENMP library. If after installing OPENMP, it still does not work,
specify the location of the OPENMP library in the makefile, for example by
adding it to LDFLAGS using the -L flag (for example,
`LDFLAGS = -L//usr/local/lib -lgomp`).

If you are desperate, feel free to email us by using the contact section below :-)


## QUICK START

You can test Zelda with a simple task by running

`./zelda params_quickstart.ini`

Zelda will perform a quick computation of the pairwise velocity statistics
from a subsample of the Millennieum simulation of side 62.5 Mpc; the catalogue
file is part of the package and is contained in the 'test_data' folder.

The result will be stored in the Zelda directory under the name
`results_millennium_small.dat`. To plot it in gnuplot, just run:

    set log x
    plot "results_millennium_small.dat" u 2:4:($4-$6/sqrt($3)):($4+$6/sqrt($3)) with yerr


## SHORT USER GUIDE

Zelda takes as input a parameter file with a list of 'key = value' settings.
The parameter file has to be text-only and usually has a non-mandatory .ini
extension. For example, you could make a test run of Zelda with

`./zelda params_quickstart.ini`

The most important file for a new user is params_explanatory.ini. It is a
parameter file with a documented list of all the parameters in Zelda.
The file can be also used as a template for creating your custom parameter files.

The directory structure of Zelda is important to learn how the code works:

* The 'source' directory contains the main source files in C. Each file
corresponds to a module in Zelda.

* The 'tool' directory contains accessory source files in C with
purely numerical functions or utility functions.

* The 'main' directory contains the main source files, i.e. the executable
files, including zelda.c.

* The 'python' directory contains Python scripts to launch Zelda, including
the Zelda wrapper (zelda.py), the batch script (zelda_script.py) and a
rebinning function.

* The 'include' directory contains the declaration files (.h) for all the
C files in the 'source', 'main' and 'tools' directories.

* The 'scripts' directory contains accessory script files in bash or 
other scripting languages. For example, to fetch catalogues from
remote servers (e.g. the Millennium simulation server) or handling
catalogue files.

* The 'test' directory contains executable programs to test the outputs
of Zelda.


## CREDITS

We wish to thank Julien Lesgourgues, Thomas Tram and Diego Blas for creating
CLASS! Without CLASS, Zelda would not exist as it uses the same modular structure
and error system.


## CONTACT

Please contact us if you need any help installing or running the code! Our
contacts are:

Guido Walter Pettinari (<guido.pettinari@gmail.com>)
Alicia Bueno Belloso (<alicia.bueno.belloso@gmail.com>)

Make sure to check Zelda's repository for news and updates:

https://github.com/coccoinomane/zelda



