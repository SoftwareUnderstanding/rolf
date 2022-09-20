RadioFisher/BAO-21cm
--------------------
Phil Bull (philbull@gmail.com)

_November 2020_

Cosmology Fisher forecasting code for HI (21cm) intensity mapping experiments
and spectroscopic galaxy surveys.

Released under the Academic Free License (AFL-3.0).

ABOUT RADIOFISHER
-----------------

RadioFisher is a Fisher forecasting code for cosmology with intensity maps of 
the redshifted 21cm emission line of neutral hydrogen. The formalism 
implemented by this code is described in Bull, Ferreira, Patel and Santos 
(2014). It's written in Python, and makes heavy use of NumPy, SciPy, and 
matplotlib. You also need CAMB. I'm running it very happily on Ubuntu 20.04 and 
other Linux machines. It should also run fine on Macs.

The code is provided openly for inspection and re-use. If you use it, please 
cite us! If you have problems getting it to work, or have bugfixes, comments, 
or suggestions, please get in touch with me. This is an actively-used scientific 
code, so expect to have to get your hands dirty!

REQUIREMENTS
------------

 - Python (tested with 3.7)
 - Recent NumPy and SciPy
 - matplotlib
 - mpi4py
 - CAMB (http://camb.info/)

INSTALLATION
------------

This is a collection of Python scripts, none of which need to be compiled or 
installed before use. However, you should compile CAMB and change the CAMB_EXEC 
variable in baofisher.py to point at this executable.

GETTING STARTED
---------------

To get started, check out the git repository and make sure you have Python 2.7 
and up-to-date versions of SciPy, NumPy, and matplotlib installed. You should 
also download and compile CAMB. It also helps if you install mpi4py too; 
although most of the Fisher forecasting code in baofisher.py doesn't need MPI, 
the full_experiment.py frontend code does. Finally, in order to use some of the 
existing experiment definitions, you’ll need to download the interferometer 
baselines package* and unpack it in the array_config/ subdirectory.
*[http://philbull.com/radiofisher_array_config.tar.gz; 5 MB]

Next, edit the CAMB_EXEC variable at the top of baofisher.py to point at the 
directory where the CAMB executable resides (not the CAMB executable itself). 
At the moment, you also need to create a subdirectory called output/.

Now, to run your first forecast, call `mpirun -n 1 ./full_experiment.py 0`. This 
will run a forecast for the first experiment listed in full_experiment.py using 
only one processor, and it will take quite a while (several minutes, usually). 
This is because the first time you run it, CAMB runs and produces a 
high-resolution P(k) for your fiducial cosmology. This is cached, so subsequent 
runs are much faster.

Output is stored in the output/ subdirectory. Look inside full_experiment.py to 
see what's saved; it's mostly a Fisher matrix per redshift bin (variable names 
are in the header of each file), and some auxiliary information about redshift 
bins, functions of redshift, and the binning of P(k) in k space.

The next time you run, try changing the first and second numbers, i.e. 
`mpirun -n <nproc> ./full_experiment.py <experiment-id>`. Each redshift bin is 
processed as a separate unit. Redshift bins are divided up between the 
available CPUs. There is no point setting <nproc> higher than the number of 
redshift bins for a given experiment.

To process the output of the forecasts, try one of the plot_*.py scripts. 
You'll probably have to edit it first to specify which experiments you want to 
see (lists of experiments are always specified near the top of the script).

WORKING WITH THE CODE
---------------------

Here are some key files that you should know about. Make sure you look inside 
them; they're heavily commented and pretty much everything has a Python 
docstring.

 * baofisher.py: Most of the forecasting code and a large number of helper 
                 functions are kept here.
 * full_experiment.py: Script for running a full forecast for a given 
                 experiment.
 * experiments.py: Specifications for a large number of experiments are defined 
                 here, as well as survey parameters and the fiducial 
                 cosmological parameters. Some of them need auxiliary baseline 
                 distribution files; these will be made public soon, but in the 
                 meantime just email me if you need them.
 * galaxy_full_experiment.py: Runs a forecast for a galaxy redshift survey 
                 rather than an IM survey.
 * plot_dlogp.py: Plots constraints on P(k) for given experiments. This is a 
                 useful first plotting script to use.
 * plot_w0wa.py: Another useful plotting script to look at, as it works with 
                 cosmological parameters rather than just P(k) on its own.

NOTES ON MPI
------------

As mentioned above, there is no point using more processors than an experiment 
has redshift bins, since the extra processors will just sit around idle.

You may find that your installation of SciPy/NumPy or CAMB causes some weird 
behaviour and hogs a lot of processor sometimes when this code is running. This 
might be caused by a conflict between MPI (which Radio Fisher uses) and OpenMP 
(which some of the libraries just mentioned have optional support for). Try 
calling the script with OMP_NUM_THREADS=2 or some other low number, e.g. 
`OMP_NUM_THREADS=2 mpirun -n 20 ./full_experiment.py 4`.

CITING THIS CODE
----------------

If you use this code in a scientific work, please cite us! The relevant paper 
is the following:

Philip Bull, Pedro G. Ferreira, Prina Patel, and Mario Santos, 
ApJ 803, 21 (2015) [arXiv:1405.1452] [doi:10.1088/0004-637X/803/1/21].

PROBLEMS / BUGS
---------------

Email me (philbull@gmail.com) with bug reports, patches, requests for features 
and so on. I'll be happy to help/fix things.

