Multi_CLASS: cross-tracer angular power spectra of number counts computed using the Cosmic Linear Anisotropy Solving System (CLASS)
==============================================

**Main developers:** Nicola Bellomo, José Luis Bernal and Giulio Scelfo

-----------------------------------------------------------------------------

Multi_CLASS is a modification of the public Boltzmann code CLASS (see information below) that allows for the computation of the **cross-tracer angular power spectra** of the number count fluctuations for two different tracers of the underlying dark matter density field. In other words, it generalizes the standard `nCl` output option of CLASS to the case of **two different tracers**, e.g., two different galaxy populations with their own redshift distribution, galaxy and magnification bias parameters, etc. 

Multi_CLASS also includes an implementation of the effect of **primordial non-Gaussianities of the local type**, parametrized by the parameter `f_NL` (following the large-scale structure convention), on the effective bias of the tracers. There is also the possibility of having a tilted non-Gaussian correction, parametrized by `n_NG`, with a pivot scale determined by `k_pivot_NG`.

Multi_CLASS already includes galaxy redshift distributions for forthcoming galaxy surveys, with the ease of choosing between them (or an input file) from the parameters input file (e.g., `multi_explanatory.ini`). In addition, Multi_CLASS includes the possibility of using resolved gravitational wave events as a tracer.

Getting started
---------------------

The installation, compilation and running of Multi_CLASS is exactly the same as for the standard CLASS code, and does not require additional libraries or dependencies. Moreover, Multi_CLASS can be used as the standard CLASS (both directly with the c executable and the python wrapper). However, the name of some parameters controlling the `nCl` output has been modified for the sake of clarity. A commented example describing all relevant or modified parameters in CLASS can be found in the file `multi_explanatory.ini`.

To check that the code runs, type:

    ./class multi_explanatory.ini

The running times of Multi_CLASS are similar to those of the standard CLASS, with the only different that Multi_CLASS computes more cross-tracer angular power spectra between redshift bins when `selection_multitracing = yes`: while for an auto-tracer correlation Cl(z_1,z_2) = Cl(z_2,z_1), this is not necessarily the case for cross-correlations between two tracers. Therefore, if `selection_multitracing = yes` is used, the number of Cls columns in the output file will be NxN instead of Nx(N+1)/2, where N is the number of redshift bins used (if cross-correlations between all redshift bins are required). The number of cross-tracer correlations between different redshift bins can be controlled as in the standard CLASS.

The same parameters input file used to compute the cross-tracer angular power spectra can be automatically used to compute the single tracer angular power spectra for the **first** tracer by using `selection_multitracing = no`.

Using the code
-------------------

You can use Multi_CLASS freely, provided that in your publications you cite the presenting papers of Multi_CLASS (where you can find more details about the code):

- [Beware of commonly used approximations I: errors in forecasts](https://arxiv.org/abs/2005.10384), Bellomo, Bernal, Scelfo, Raccanelli and Verde; JCAP **10** (2020) 016.
- [Beware of commonly used approximations II: estimating systematic biases in the best-fit parameters](https://arxiv.org/abs/2005.09666), Bernal, Bellomo, Raccanelli and Verde; JCAP **10** (2020) 017.

And, of course, cite the original CLASS papers, at least:

- [The Cosmic Linear Anisotropy Solving System (CLASS) II: Approximation schemes](http://arxiv.org/abs/1104.2933), Blas, Lesgourgues and Tram (2011).

-------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------


CLASS: Cosmic Linear Anisotropy Solving System 
==============================================

Authors: Julien Lesgourgues and Thomas Tram

with several major inputs from other people, especially Benjamin
Audren, Simon Prunet, Jesus Torrado, Miguel Zumalacarregui, Francesco
Montanari, etc.

For download and information, see http://class-code.net


Compiling CLASS and getting started
-----------------------------------

(the information below can also be found on the webpage, just below
the download button)

Download the code from the webpage and unpack the archive (tar -zxvf
class_vx.y.z.tar.gz), or clone it from
https://github.com/lesgourg/class_public. Go to the class directory
(cd class/ or class_public/ or class_vx.y.z/) and compile (make clean;
make class). You can usually speed up compilation with the option -j:
make -j class. If the first compilation attempt fails, you may need to
open the Makefile and adapt the name of the compiler (default: gcc),
of the optimization flag (default: -O4 -ffast-math) and of the OpenMP
flag (default: -fopenmp; this flag is facultative, you are free to
compile without OpenMP if you don't want parallel execution; note that
you need the version 4.2 or higher of gcc to be able to compile with
-fopenmp). Many more details on the CLASS compilation are given on the
wiki page

https://github.com/lesgourg/class_public/wiki/Installation

(in particular, for compiling on Mac >= 10.9 despite of the clang
incompatibility with OpenMP).

To check that the code runs, type:

    ./class explanatory.ini

The explanatory.ini file is THE reference input file, containing and
explaining the use of all possible input parameters. We recommend to
read it, to keep it unchanged (for future reference), and to create
for your own purposes some shorter input files, containing only the
input lines which are useful for you. Input files must have a *.ini
extension.

If you want to play with the precision/speed of the code, you can use
one of the provided precision files (e.g. cl_permille.pre) or modify
one of them, and run with two input files, for instance:

    ./class test.ini cl_permille.pre

The files *.pre are suppposed to specify the precision parameters for
which you don't want to keep default values. If you find it more
convenient, you can pass these precision parameter values in your *.ini
file instead of an additional *.pre file.

The automatically-generated documentation is located in

    doc/manual/html/index.html
    doc/manual/CLASS_manual.pdf

On top of that, if you wish to modify the code, you will find lots of
comments directly in the files.

Python
------

To use CLASS from python, or ipython notebooks, or from the Monte
Python parameter extraction code, you need to compile not only the
code, but also its python wrapper. This can be done by typing just
'make' instead of 'make class' (or for speeding up: 'make -j'). More
details on the wrapper and its compilation are found on the wiki page

https://github.com/lesgourg/class_public/wiki

Plotting utility
----------------

Since version 2.3, the package includes an improved plotting script
called CPU.py (Class Plotting Utility), written by Benjamin Audren and
Jesus Torrado. It can plot the Cl's, the P(k) or any other CLASS
output, for one or several models, as well as their ratio or percentage
difference. The syntax and list of available options is obtained by
typing 'pyhton CPU.py -h'. There is a similar script for MATLAB,
written by Thomas Tram. To use it, once in MATLAB, type 'help
plot_CLASS_output.m'

Developing the code
--------------------

If you want to develop the code, we suggest that you download it from
the github webpage

https://github.com/lesgourg/class_public

rather than from class-code.net. Then you will enjoy all the feature
of git repositories. You can even develop your own branch and get it
merged to the public distribution. For related instructions, check

https://github.com/lesgourg/class_public/wiki/Public-Contributing

Using the code
--------------

You can use CLASS freely, provided that in your publications, you cite
at least the paper `CLASS II: Approximation schemes <http://arxiv.org/abs/1104.2933>`. Feel free to cite more CLASS papers!

Support
-------

To get support, please open a new issue on the

https://github.com/lesgourg/class_public

webpage!
