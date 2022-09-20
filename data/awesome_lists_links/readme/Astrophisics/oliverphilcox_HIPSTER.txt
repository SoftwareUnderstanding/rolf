# HIPSTER

## HIgh-k Power Spectrum EstimatoR

Code to compute small-scale power spectra and isotropic bispectra for cosmological simulations and galaxy surveys of arbitrary shape, based on the configuration space estimators of Philcox & Eisenstein (2019, MNRAS, [arXiv](https://arxiv.org/abs/1912.01010)) and Philcox (2020, submitted, [arXiv](https://arxiv.org/pdf/2005.01739.pdf)). This computes the Legendre multipoles of the power spectrum, <img src="/tex/a939b8abbd34a6a7097130a860c9ebc2.svg?invert_in_darkmode&sanitize=true" align=middle width=38.738704949999985pt height=24.65753399999998pt/> or bispectrum, <img src="/tex/228c84969208bf099519c5405ba50503.svg?invert_in_darkmode&sanitize=true" align=middle width=70.74890294999999pt height=24.65753399999998pt/> by computing weighted pair counts over the simulation box or survey, truncated at some maximum radius <img src="/tex/12d208b4b5de7762e00b1b8fb5c66641.svg?invert_in_darkmode&sanitize=true" align=middle width=19.034022149999988pt height=22.465723500000017pt/>. This fully accounts for window function effects, does not include shot-noise or aliasing, and is optimized for small-scale spectrum computation in real- or redshift-space. Both the power spectrum and bispectrum estimators have efficiency <img src="/tex/c7848231674242334f6e18154b76a7ac.svg?invert_in_darkmode&sanitize=true" align=middle width=53.726103749999986pt height=27.94539330000001pt/> for <img src="/tex/f9c4988898e7f532b9f826a75014ed3c.svg?invert_in_darkmode&sanitize=true" align=middle width=14.99998994999999pt height=22.465723500000017pt/> particles, and become faster at large <img src="/tex/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode&sanitize=true" align=middle width=9.075367949999992pt height=22.831056599999986pt/>.

The code can be run either in 'aperiodic' or 'periodic' mode, for galaxy surveys or cosmological simulations respectively. The 'periodic' mode contains various optimizations relating to the periodic geometry, as detailed in the second paper. HIPSTER also supports weighted spectra, for example when tracer particles are weighted by their mass in a multi-species simulation. Generalization to anisotropic bispectra is straightforward (and requires no additional computing time) and can be added on request.

Full documentation of HIPSTER is available on [ReadTheDocs](https://HIPSTER.readthedocs.io).

### Basic Usage

To compute a power spectrum from particles in a *periodic* simulation box (``data.dat``) up to <img src="/tex/720b52da688c892f252bc47ce206b36d.svg?invert_in_darkmode&sanitize=true" align=middle width=39.95424014999999pt height=22.831056599999986pt/> with pair-counts truncated at radius <img src="/tex/12d208b4b5de7762e00b1b8fb5c66641.svg?invert_in_darkmode&sanitize=true" align=middle width=19.034022149999988pt height=22.465723500000017pt/> using <img src="/tex/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode&sanitize=true" align=middle width=9.075367949999992pt height=22.831056599999986pt/>-space binning file ``binning.csv`` and 4 CPU cores, run:

    ./hipster_wrapper_periodic.sh --dat data.dat --l_max L -R0 R0 -k_bin binning.csv --nthreads 4

To compute a power spectrum from galaxies in a *non-periodic* survey (``data.dat``), defined by a set of randoms (``randoms.dat``), up to <img src="/tex/720b52da688c892f252bc47ce206b36d.svg?invert_in_darkmode&sanitize=true" align=middle width=39.95424014999999pt height=22.831056599999986pt/>, truncating pair-counts at <img src="/tex/12d208b4b5de7762e00b1b8fb5c66641.svg?invert_in_darkmode&sanitize=true" align=middle width=19.034022149999988pt height=22.465723500000017pt/> and using <img src="/tex/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode&sanitize=true" align=middle width=9.075367949999992pt height=22.831056599999986pt/>-space binning file ``binning.csv``, with 4 CPU-cores, run:

    ./hipster_wrapper.sh --dat galaxies.dat --ran_DR randoms.dat --ran_RR randoms.dat -l_max L -R0 R0 -k_bin binning.csv --nthreads 4

To compute an isotropic bispectrum from particles in a *periodic* simulation box (``data.dat``) up to <img src="/tex/720b52da688c892f252bc47ce206b36d.svg?invert_in_darkmode&sanitize=true" align=middle width=39.95424014999999pt height=22.831056599999986pt/> with pair-counts truncated at radius <img src="/tex/12d208b4b5de7762e00b1b8fb5c66641.svg?invert_in_darkmode&sanitize=true" align=middle width=19.034022149999988pt height=22.465723500000017pt/> using <img src="/tex/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode&sanitize=true" align=middle width=9.075367949999992pt height=22.831056599999986pt/>-space binning file ``binning.csv`` and 4 CPU cores, using 3 times as many random points as data points, run:

        ./hipster_wrapper_bispectrum.sh --dat data.dat --l_max L -R0 R0 -k_bin binning.csv --nthreads 4 --f_rand 3

For any queries regarding the code please contact [Oliver Philcox](mailto:ohep2@cantab.ac.uk).

**New for version 2**: Optimizations for periodic N-body simulations

**New for version 3**: A new pair-count estimator for the periodic bispectrum
