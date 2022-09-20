# hfs_fit
**A python fitting program for atomic emission lines with hyperfine structure (HFS).**

**It has nice sliders for parameters to change the fit visually, useful when HFS constants are unknown.**

<img src="data/z3P2---a3D3 (spectrum.txt).png">

# Quickstart

* requires python 3.6 or higher
* run `pip install -rrequirements.txt` to install dependencies
* run `example.py` in an interactive python environment

# Files and Explanations

1) example.py - basic usage.

2) fitLog.xlsx - parameters saved here when desired.

3) hfs_fit.py - Main script that makes use of others, contains class for spectrum, contains fitting, plotting algorithms.

4) interpolation.py - used for cubic spline interpolation when specified in hfs_fit.py.

5) LU.py - LU decomposition for interpolation.py.

6) matrixmult.py - matrix multiplication for LU.py.

7) relInt.py - routine to calculate relative intensities of HFS components, used by hfs_fit.py.

8) spectrum.txt - a small portion of an UV Co II spectrum with 4 Co II lines.

9) fits - folder containing saved plots.

# Useful Functions and Notes

- Can plot transition diagram with components using the LineFig() method in the hfs class. nInterp argument for this is the number of points to artificially add to make lines smooth, 1 for no interpolation (default). The spacing between texts may not be perfect, most of the time the level label will touch a level line, can change this by changing the location of the texts from lines 678-681. 

- Can plot spectrum using the PlotSpec() method in the hfs class, put a wavenumber in the bracket and it will plot around that wavenumber.

- Can plot residual using Residual() in the hfs class, e.g. class.Residual(class.paramsGuess,plot = True)

- Use hjw() of hfs class to half all jumpwidths before Optimise(), this is convenient when performing the final optimisation of parameters, or if the initial guess is very good.

- Can always re-open the sliders plot with PlotGuess() method of the hfs class. If the sliders don't work, try closing and opening it up again (this happens sometimes in iPython).

- Can also add points for smoothing during fitting, to do this change the nInterp value in the WNRange() method of hfs and re-import hfs.

- HFS components are plotted by default, can turn this off using PlotGuess(components = False)

- The reset button of PlotGuess() doesn't seem to work in iPython.

- If the instrumental profile (Fourier transform spectroscopy only) is negligible, put icut at the maximum value.

