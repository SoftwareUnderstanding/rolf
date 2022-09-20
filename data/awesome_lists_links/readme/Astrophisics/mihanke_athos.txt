# ATHOS
## On-the-fly stellar parameter determination of FGK stars based on flux ratios from optical spectra

![alt text](https://raw.githubusercontent.com/mihanke/athos/master/ATHOS_logo.png)

*M. Hanke, C. J. Hansen, A. Koch, and E. K. Grebel*

__NEW function added: ATHOS can now analyze spectra with resolutions >> 45000 by convolution with an appropriate Gaussian kernel.__

ATHOS (__A__ __T__ ool for __HO__ mogenizing __S__ tellar parameters) is __A__ (non-exhaustive, users are encouraged to adapt the tool to their needs!) computational implementation of the spectroscopic stellar parameterization method outlined in [Hanke et al. (2018)](https://www.aanda.org/articles/aa/full_html/2018/11/aa33351-18/aa33351-18.html). Once configured properly, it will measure flux ratios in the input spectra and deduce the stellar parameters *effective temperature*, *iron abundance* (a.k.a [Fe/H]), and *surface gravity* by employing pre-defined analytical relations. The code is written in Python and has been tested to work properly with Python 2.7+ and Python 3.4+. ATHOS can be configured to run in parallel in an arbitrary number of threads, thus enabling the fast and efficient analysis of huge datasets. 

Requirements
---

* `python` 2.7+ or 3.4+
* `numpy` 1.12+
* `astropy` 2.0+
* `pandas` 0.22+
* `multiprocessing` 0.7+
* `joblib` 0.12+

Input data
---
The routines are designed to deal with *__one-dimensional, optical__* stellar spectra that are *__shifted to the stellar rest frame__* (see [__PAPER__](https://www.aanda.org/articles/aa/full_html/2018/11/aa33351-18/aa33351-18.html) for details). ATHOS supports several types of file structures, among which are standard 1D fits spectra, fits binary tables, numpy arrays, and plain text (see function `athos_utils.read_spectrum` for details).

Usage
===
In order to execute ATHOS, copy the files `athos.py` and `athos_utils.py`, as well as the folder `/coefficients` to the same local directory. After modifying the parameter file `parameters.py` (see next section), the code can be run from terminal by executing

```bash
$ python athos.py
```
every time ATHOS is used, or by making `athos.py` executable
```bash
$ chmod +x athos.py
```
once and running it using
```bash
$ ./athos.py
```
on every subsequent call.

The parameter file `parameters.py`
---
ATHOS is initialized via parameters read from a file called `parameters.py`, which should be located in the same directory as `athos.py`. An example follows below (included in this repository):

```python
input_specs = 'path/to/input/file'  # A string pointing to the file with information about the input spectra
output_file = 'path/to/output/file' # A string specifying the desired output file 
dtype = 'lin'         # one of 'lin', 'log10', or 'ln'
wunit = 'aa'          # either 'aa' or 'nm'
R = 45000             # The instrumental (or, in case of higher rotation, effective) resolution
tell_rejection = True # Either True or False. If True, the range lambda - lamda_i/R < lambda < lambda + lambda_i/R will be masked for each internally stored telluric lambda_i
n_threads = -1        # number of threads for parallelization; all available cores/threads if set to -1
# wave_keywd = None   # Wavelength keyword for fits binary table spectra.
# flux_keywd = None   # Flux keyword for fits binary table spectra.
# verbose = True      # Provide all measurements from individual FRs
```
The following seven (nine) parameters must be set:
* `input_specs`: A string containing the (absolute or relative) path to the input text file that stores the information about the spectra (see next section).
* `output_file`: A string telling ATHOS where to save the output results.
* `dtype`: A string denoting the dispersion type of the input spectra. Valid options are 'lin', 'log10', or 'ln'.
* `wunit`: The wavelength unit can either be Angstroms ('aa') or nanometers ('nm').
* `R`: The resolution of the spectrograph. For stars with substantial rotation (*v*sin*i* > 5 km/s), an effective resolution R = 1/sqrt(1/Rinst^2 + (*v*sin*i*/c)^2) should be provided.
* `tell_rejection`: A flag specifying whether telluric rejection should be performed. If `tell_rejection` is set to `True`, the relative velocity of the topocenter has to be provided in the file `input_specs` (see next section).
* `n_threads`: The number of threads used for parallel computation. A value of `-1` indicates that all available cores/threads should be used.
* `wave_keywd` (optional): A string telling ATHOS where to look for the wavelength information in a fits binary table. It should be set to `None`, commented out, or completely deleted if the input spectra are not in fits binary table format. Further, `wave_keywd` does not need to be explicitly set if it is `WAVE`.
* `flux_keywd` (optional): A string telling ATHOS where to look for the flux information in a fits binary table. It should be set to `None`, commented out, or completely deleted if the input spectra are not in fits binary table format. Further, `flux_keywd` does not need to be explicitly set if it is `FLUX` or `FLUX_REDUCED`.
* `verbose` (optional): A boolean controlling whether ATHOS will provide an additional file (`output_file` + '\_verbose') containing the measurements from individual FRs in the order of appearance in tables A.1, A.2, and A.3 of the [__paper__](https://www.aanda.org/articles/aa/full_html/2018/11/aa33351-18/aa33351-18.html).

The input file `input_specs`
---
Each line in the text file `input_specs` should contain the information about one input spectrum. Columns must be separated by whitespace. Comment lines must begin with a `#`. All other lines should obey the following structure:
* __1st column__: The absolute or relative path to the spectrum file.
* __2nd column__ (optional): The relative velocity of the topocenter, `v_topo`, in km/s. The sign convention is such that if the input spectrum is blue-shifted w.r.t. the topocentric rest frame `sign(v_topo) = +1` applies, and `sign(v_topo) = -1` otherwise. This parameter is only relevant if `tell_rejection=True`.
* __subsequent columns__ (optional): In the following `n` columns, polynomial coefficients for polynomials of degree `n-1` as a function of wavelength (in Angstroms) can be provided. These are used to weigh FRs from different parts of the input spectrum, which may be desirable due to, e.g., strong extinction in the bluer portions. The coefficients must follow the order from highest (`n-1`) to lowest (`0`) degree. In case weights are provided, the final parameters are computed from their individual measurements through the weighted median. 

A minimal example for `input_specs`:
```
/home/user/pathtofile/spectrum.fits
```
A more complex example:
```
# This file tells ATHOS to parameterize the spectra '/home/user/pathtofile/spectrum.fits' and 
# '/home/user/pathtofile/spectrum1.fits'. 
#
# The velocity of the topocenter is 21.4 km/s and -12.7 km/s, respectively.
#
# For the second spectrum, each individual measurement x_i obtained at wavelength lambda_i for the parameter x 
# will be assigned the weight w_i = P(lambda_i)/sum_n(P(lambda_n)) with 
# P(lambda_i) = 1.24e-2 * lambda_i ** 2 + 0.0 * lambda_i + 0.1.
#
# spectra                            v_topo  p_n-1    p_n-2  p_n-3 [...]
/home/user/pathtofile/spectrum.fits   21.4
/home/user/pathtofile/spectrum1.fits -12.7   1.24e-2  0.0    0.1
# Input lines with preceding '#' are ignored.
```

The output file `output_file`
---
After successful completion, the code will produce the text file `output_file` (defined in `parameters.py`) containing the following columns:
* `spectrum`: The input spectra.
* `Teff`, `Teff_err_stat`, and `Teff_err_sys`: The (weighted) median temperatures, their median absolute deviation, and the propagated systematic errors.  
* `[Fe/H]`, `[Fe/H]_err_stat`, and `[Fe/H]_err_sys`: The (weighted) median [Fe/H], their median absolute deviation, and the propagated systematic errors.  
* `logg`, `logg_err_stat`, and `logg_err_sys`: The (weighted) median logg, their median absolute deviation, and the propagated systematic errors.

An output for the above (2nd) example for `input_specs` could look as follows:
```
#spectrum Teff Teff_err_stat Teff_err_sys [Fe/H] [Fe/H]_err_stat [Fe/H]_err_sys logg logg_err_stat logg_err_sys
/home/user/pathtofile/spectrum.fits    4000    58    97  -2.14   0.07  0.16  1.83  0.19  0.36
/home/user/pathtofile/spectrum1.fits   6300   182    97   0.21   0.23  0.17  4.37  0.28  0.36
```

Notes
---
The following issues will be resolved in the very near future:
* In the current release, flux errors (or S/N values) have no effect on the final statistical parameter errors, which are solely based on the median absolute deviation of the ensemble of measurements for each parameter.
* So far, no error or exception handlings beyond those implemented in the external modules are available. Consequently, inappropriate input might raise rather cryptic errors/exceptions.

References
---
"ATHOS: On-the-fly stellar parameter determination of FGK stars based on flux ratios from optical spectra"
M. Hanke, C. J. Hansen, A. Koch, E. K. Grebel, [2018, A&A, 619, A134](https://www.aanda.org/articles/aa/full_html/2018/11/aa33351-18/aa33351-18.html) ([arXiv:1809.01655v1](https://arxiv.org/abs/1809.01655))
