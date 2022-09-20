# spinOS: the SPectroscopic and INterferometric Orbital Solution finder.

[![GitHub license](https://img.shields.io/github/license/matthiasfabry/spinOS)](https://github.com/matthiasfabry/spinOS/blob/master/COPYING.txt)
[![Other](https://img.shields.io/badge/ASCL-2102.001-blue)](https://ascl.net/2102.001)
![Github version](https://img.shields.io/badge/version-v2.7.3-red)

## Goal:

spinOS calculates binary orbital elements. Given a set of radial velocity measurements of a
spectroscopic binary and/or relative position measurement of an astrometric binary, spinOS fits an
orbital model by minimizing a chi squared metric. You need to supply an initial guess for the
parameters that define the binary orbit:

- p:       the period of the binary
- e:       the eccentricity of the orbit
- i:       the inclination of the orbit (with respect to the plane of the sky)
- omega:   the argument of periastron of the secondary, with respect to its ascending node
- Omega:   the longitude of the ascending node of the secondary, measured East from North
- t0:      the time of periastron passage
- d:       the distance to the system
- k1:      the semiamplitude of the RV curve of the primary
- k2:      the semiamplitude of the RV curve of the secondary
- gamma1:  the peculiar velocity of the primary
- gamma2:  the peculiar velocity of the secondary
- mt:      the total mass of the system

This package provides a GUI to easily visualize your data. The GUI allows for easy plotting of data
and models, as well as minimization of the model to your supplied data. The program then gives a
best fit value for the parameters itemized above.

## Usage:

To use the GUI, simply run:

    python spinOS.py [dir]

In the GUI data tab, put your working directory, where all your data files are relative to where you
launched spinOS from. You can select files to be loaded in, or add Datasets and data entries
manually in the respective tabs of the Data Manager. When loading in data, see formatting below.

In the System/parameters tab, you can play with the systemic parameters, and see the changes of the
orbit on the plots by pressing 'refresh'. With the checkbuttons, indicate which parameters should be
minimized. Below, some inferred parameters of the model are presented. Use the load guesses button
to load all guesses from your guessfile indicated in the data tab. The save buttons save either the
guesses to files you specify in the output names tab (warning: may overwrite!)

In the minimize tab, you can apply a custum weighting of the astrometric data to the chi-squared
value (typically, you would want to increase this if you trust astrometry better than spectroscopy,
and have only few astrometric measurements). You can minimize the model to the selected data with
the minimize button, selecting a method first. Levenberg-Marquardt does local non-linear least
squares minimization, while basinhopping tries to find a global minimum by iterating local
Nelder-Mead simplex minimizations. Note that basinhopping can take considerably longer with lots of
free parameters. Alternatively, you can select MCMC which first does a local LM minimization
followed by an mcmc error estimation. When MCMC sampling, the first _burn_ (default = 100)
samples are discarded, and then only 1 every _thin_ (default = 1) samples are retained in the final
results. The philosophy behind this is that the underlying sampler does not draw independent samples
from the posterior distribution, it first needs to 'settle' to the maximum likelihood region (hence
the burning), and then a random walk will only yield independent results twice every time the
characteristic autocorrelation time has passed (hence the thinning). These parameters are difficult
to estimate beforehand. When minimizing, error are estimated as the diagonal elements of the
correlation matrix, or as half of the difference between the 15.87 and 84.13 percentiles found in an
Markov Chain Monte Carlo sampling if you selected MCMC.

In the plot controls tab, various checkbuttons are provided to plot certain elements on the plot
windows on the right. The phase slider allows for overplotting a dot at the phase indicated (for
illustrative purposes, eg, for visually connecting the apparent orbit with the RV plot).

### Data formatting:

This application expects the data to be in the following format: All data files should be plain text
files, formatted as:
for RV data:

    JD(days) RV(km/s) error_on_RV(km/s)

_e.g._:

    45000 25.1 2.1
    45860 -4.2 1.1
    etc...

for AS data:
either (set button E/N):

    JD(days) E_separation(mas) N_separation(mas) semimajor_ax_errorellipse(mas) semiminor_ax_errorellipse(mas) angle_E_of_N_of_major_ax(deg)

_e.g._:

    48000 -2.5 2.4 0.1 0.8 60
    48050 2.1 8.4 0.4 0.5 90
    etc...

or (set button Sep/PA):

    JD(days) separation(mas) PA(deg) semimajor_ax_errorellipse(mas) semiminor_ax_errorellipse(mas) angle_E_of_N_of_major_ax(deg)

_e.g._:

    48000 3.5 316 0.1 0.8 60
    48050 8.7 76 0.4 0.5 90
    etc...

for the guess file, format should be _e.g._:

    e 0.648 True
    i 86.53 True
    omega 211.0 True
    Omega 67.3 True
    t0 56547.1 True
    k1 31.0 False
    k2 52.0 True
    p 3252.0 True
    gamma1 15.8 False
    gamma2 5.6 False
    mt 30.0 True

All eleven parameters should be guessed if you load in guesses. (for their meaning see above)

## Dependencies:

    python 3.9.10
    tk 8.6.12
    numpy 1.22.2
    scipy 1.8.0
    lmfit 1.0.3
    matplotlib 3.5.1
    emcee 3.1.1 (if MCMC error calculation is performed)
    corner 2.2.1 (if MCMC corner diagram is plotted, you need pandas for this too)
    A LaTeX distribution that allows matplotlib.rc(usetex=True)
    the supplied yaml file can be used to configure your python evironment

## Author:

Matthias Fabry  
Instituut voor Sterrekunde, KU Leuven, Belgium

## Date:

17 January 2022

## Licence:

Copyright 2020, 2021, 2022 Matthias Fabry. This software is released under the GNU GPL-3.0-or-later
License.

## Acknowledgements:

We thank the authors of lmfit for the development of their package.