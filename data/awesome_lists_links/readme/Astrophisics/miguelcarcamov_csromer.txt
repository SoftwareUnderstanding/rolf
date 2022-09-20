# CS-ROMER

[![codecov](https://codecov.io/gh/miguelcarcamov/csromer/branch/development/graph/badge.svg?token=QRRJNFSF7P)](https://codecov.io/gh/miguelcarcamov/csromer)

*Compressed Sensing ROtation MEasure Reconstruction*

Compressed sensing reconstruction framework for Faraday depth spectra.
Please feel free to open an issue if you spot a bug. This is an open source project, and therefore you can fork, make changes and submit a [pull request](https://github.com/miguelcarcamov/csromer/pulls) of any of your additions and modifications.

- This paper explains what is [Faraday rotation measure synthesis](https://www.aanda.org/articles/aa/abs/2005/39/aa2990-05/aa2990-05.html)
- Wikipedia information about [Faraday effect](https://en.wikipedia.org/wiki/Faraday_effect)

## Features

- Simulation of Faraday depth sources
- Subtraction of Galactic RM
- Reconstruction of Faraday depth sources from linearly polarized data
- Reconstruction of Faraday depth sources using Compressed Sensing
- More than 100 wavelet filters provided by `Pywavelets`

This code will run in a Python >= 3.9.7 environment with all the packages installed (see `requirements.txt` file).

## Citing

The paper of this software is under submission but if you use it you can cite it as:

```tex
@misc{https://doi.org/10.48550/arxiv.2205.01413,
  doi = {10.48550/ARXIV.2205.01413},

  url = {https://arxiv.org/abs/2205.01413},

  author = {Cárcamo, Miguel and Scaife, Anna M. M. and Alexander, Emma L. and Leahy, J. Patrick},

  keywords = {Instrumentation and Methods for Astrophysics (astro-ph.IM), Astrophysics of Galaxies (astro-ph.GA), FOS: Physical sciences, FOS: Physical sciences},

  title = {CS-ROMER: A novel compressed sensing framework for Faraday depth reconstruction},

  publisher = {arXiv},

  year = {2022},

  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

## Installation

The software can be installed as a python package locally or using Pypi

### Locally after cloning the project

```shell
git clone https://github.com/miguelcarcamov/csromer.git
cd csromer
pip install .
```

### Locally as developer

```shell
git clone git@github.com:miguelcarcamov/csromer.git
cd csromer
pip install -e .
```

We highly recommend installing [pre-commit](https://pre-commit.com) to develop over this code.
This will allow you to run hooks that reformat the project files according to our style.

### From PyPI

`pip install csromer`

### From Github

`pip install -U git+https://github.com/miguelcarcamov/csromer.git`

### From latest docker container

`docker pull ghcr.io/miguelcarcamov/csromer:latest`

## Simulate Faraday sources directly in frequency space

CS-ROMER is able to simulate Faraday depth spectra directly in wavelength-squared space. The classes `FaradayThinSource` and `FaradayThickSource` inherit directly from `Dataset`, and therefore you can directly use them as an input to your reconstruction.

### Thin sources

```python
import numpy as np
from csromer.simulation import FaradayThinSource
# Let's create an evenly spaced frequency vector from 1.008 to 2.031 GHz (JVLA setup)
nu = np.linspace(start=1.008e9, stop=2.031e9, num=1000)
# Let's say that the peak polarized intensity will be 0.0035 mJy/beam with a spectral index = 1.0
peak_thinsource = 0.0035
# The Faraday source will be positioned at phi_0 = -200 rad/m^2
thinsource = FaradayThinSource(nu=nu, s_nu=peak_thinsource, phi_gal=-200, spectral_idx=1.0)
```

### Thick sources

```python
import numpy as np
from csromer.simulation import FaradayThickSource
# Let's create an evenly spaced frequency vector from 1.008 to 2.031 GHz (JVLA setup)
nu = np.linspace(start=1.008e9, stop=2.031e9, num=1000)
# Let's say that the peak polarized intensity will be 0.0035 mJy/beam with a spectral index = 1.0
peak_thicksource = 0.0035
# The Faraday source will be positioned at phi_0 = 200 rad/m^2 and will have a width of 140 rad/m^2
thicksource = FaradayThickSource(nu=nu, s_nu=peak_thicksource, phi_fg=140, phi_center=200, spectral_idx=1.0)
```

### Simulate

Once you have set your source parameters, you can call the `simulate()` function as

```python
thinsource.simulate()
thicksource.simulate()
```

This call will simulate the linealy polarized emission and it will assign the data to the `data` attribute.

### Mixed sources

A thin+thick or mixed source is simply a superposition/sum of a thin source and thick source. Therefore we have overriden the `+` operator in order to sum these two objects.

```python
mixedsource = thinsource + thicksource
```

The result will be a `FaradaySource` object.

### Remove frequency channels randomly as if you were doing RFI flagging

The framework also allows you to randomly remove data with the function `remove_channels` to simulate RFI flagging

```python
# Let's say that we want to randomly remove 20% of the data
mixedsource.remove_channels(0.2)
```

### Adding noise to your simulations

If we want to add random Gaussian noise to our simulation we can simply call the function `apply_noise`

```python
# Let's add Gaussian random noise with mean 0 and standard deviation equal
# to 20% the peak of the signal.
sigma = 0.2*mixedsource.s_nu
mixedsource.apply_noise(sigma)
```

## Reconstruct 1D Faraday sources

To illustrate how to reconstruct Faraday depth signals with CS-ROMER first we will reconstruct the mixed source that we have just constructed

### Dirty Faraday depth spectra

```python
from csromer.reconstruction import Parameter
from csromer.transformers import DFT1D
# We first need to initialize the parameter object that will contain our Faraday depth
# data either in Faraday-depth space or in wavelet space
parameter = Parameter()
# We calculate the cellsize in Faraday depth space using an oversampling factor of 8
# Here parameter.data is set as a complex array of zeros
parameter.calculate_cellsize(dataset=mixedsource, oversampling=8)
# We instantiate our discrete Fourier transform
dft = DFT1D(dataset=mixedsource, parameter=parameter)
# We calculate the dirty Faraday depth spectra
F_dirty = dft.backward(mixedsource.data)
```

### Reconstruct simulated data

```python
from csromer.transformers import NUFFT1D
# We instantiate our non-uniform FFT
nufft = NUFFT1D(dataset=mixedsource, parameter=parameter, solve=True)
# At this point we can use either the parameter data set with zeros or we can
# use the dirty Faraday depth spectra
parameter.data = F_dirty
parameter.complex_data_to_real() # We convert the complex data to real
# You can set the L1 lambda regularization manually or estimate it as
lambda_l1 = np.sqrt(mixedsource.m + 2*np.sqrt(mixedsource.m)) * np.sqrt(2) * np.mean(mixedsource.sigma)
```

### Objective function

```python
from csromer.objectivefunction import L1, Chi2
from csromer.objectivefunction import OFunction
# We instantiate each part of our objective function
chi2 = Chi2(dft_obj=nufft, wavelet=None) # chi-squared
l1 = L1(reg=lambda_l1) # L1-norm regularization

F_obj = OFunction([chi2, l1]) # Whole objective function
f_obj = OFunction([chi2]) # Only chi-squared
g_obj = OFunction([l1]) # Just regularizations
```

### Optimization algorithm

One of the ways to optimize the objective function is to use the FISTA algorithm.

```python
from csromer.optimization import FISTA
# We instantiate our FISTA object as
opt = FISTA(guess_param=parameter, F_obj=F_obj, fx=chi2, gx=g_obj, noise=mixedsource.theo_noise, verbose=False)
# We run the optimization algorithm
obj, X = opt.run()
X.real_data_to_complex() # We convert the data back to complex when the optimization finishes
```

This returns the objective function value `obj` and `X`a `Parameter` instance object. Therefore in this case `X.data` will hold the reconstructed Faraday depth spectra.
At this point you can also access to the model and residual data in wavelength-squared as `mixedsource.model_data` and `mixedsource.residual`, respectively. You can calculate the residuals in Faraday depth space by using the DFT object as

```python
F_residual = dft.backward(mixedsource.residual)
```

### Using discrete or undecimated wavelets

CS-ROMER has about 100 filters to user with discrete wavelet transforms or undecimated wavelet transforms. We use the `Pywavelets` package, for more information please refer to [PyWavelets](https://pywavelets.readthedocs.io/en/latest/index.html). To use the wavelets in cs-romer you can do:

```python
from csromer.dictionaries import DiscreteWavelet, UndecimatedWavelet
# This line instantiates a discrete wavelet
wav = DiscreteWavelet(wavelet_name="coif3", mode="periodization", append_signal=False)
# This line instantiates an undecimated wavelet
wav = UndecimatedWavelet(wavelet_name="sym2", mode="periodization", append_signal=True)
```

The `append_signal` parameter plugs the Faraday depth spectrum to your coefficients resulting in redundancy in your coefficients. If you just want the wavelet coefficients then set `append_signal=False`.
At this point our parameter object data needs to be our coefficients and not our Faraday depth spectra, therefore, we do

```python
parameter.data = F_dirty # Suppose that you set your parameter data with your dirty Faraday depth spectrum
parameter.complex_data_to_real() # We convert the data to real
# Here we do a wavelet decomposition of our Faraday depth space
# We set the coefficients of the decomposition as our parameter data
parameter.data = wav.decompose(parameter.data)
# Don't forget to change your chi-squared
chi2 = Chi2(dft_obj=nufft, wavelet=wav)
```

You might have noticed that at the end of the optimization we will end up with fitted coefficients instead of a Faraday depth spectrum.
Therefore, we need to reconstruct the Faraday depth spectrum from our coefficients doing

```python
X.data = wav.reconstruct(X.data) # We reconstruct the Faraday depth spectrum from coefficients
X.real_data_to_complex() # We convert the real Faraday depth spectrum into complex
```

### Reconstruct a real line of sight data

To reconstruct real data your main script should follow the same workflow. The only difference is that you need to instantiate a `Dataset` object.

```python
from csromer.base import Dataset
# nu is the irregular spaced frequency
# data is the polarized emission
# sigma is the error per channel (this can be an array of ones or rms calculation per image channel)
# alpha is the spectral index at this line of sight
dataset = Dataset(nu=nu, data=data, sigma=sigma, spectral_idx=alpha)
```

### Subtracting the Milky Way RM contribution

We use [S. Hutschenreuter et al.](https://www.aanda.org/articles/aa/full_html/2022/01/aa40486-21/aa40486-21.html) Faraday sky HealPIX image to subtract the galactic RM contribution at a certain position of the sky using the object `FaradaySky`.
Note that you can omit this step, and subtract any RM value that you might find appropiate.

```python
from csromer.faraday_sky import FaradaySky
from astropy.coordinates import SkyCoord
import astropy.units as un

f_sky = FaradaySky()
coord = SkyCoord(ra=173.694*un.deg, dec=48.957*un.deg, frame="fk5")
gal_mean, gal_std = f_sky.galactic_rm(coord.ra, coord.dec, frame="fk5")
dataset.subtract_galacticrm(gal_mean.value)
```

## Reconstruct a cube

We warn the users that not all framework functions are yet implemented to work with data cubes. Therefore, we need to use `numpy` broadcasting and the package `joblib`. Let's say that you have read your polarized cube and frequency array using `np.load`. For this example we will assume that you will reconstruct with uniform weights.

```python
import numpy as np
from csromer.reconstruction import Parameter
from csromer.base import Dataset
from joblib import Parallel, delayed

QU_cubes = np.load('qu_cubes.npy') # Shape (freqs, m, n)
nu = np.load('nu.npy') # Shape (freqs,)
m = QU_cubes.shape[1]
n = QU_cubes.shape[2]

Q = QU_cubes[0]
U = QU_cubes[1]
data = Q + 1j * U
sigma = np.ones_like(nu) # Uniform weights
# We will construct a dataset only to obtain Faraday-space array shapes
foo_dataset = Dataset(nu=nu, sigma=sigma, spectral_idx=0.0)
foo_parameter = Parameter()
parameter.calculate_cellsize(dataset=foo_dataset, oversampling=8)
# Faraday dispersion function cube
# Note that ee add another dimension to store dirty, model, residual and restored signals
F = np.zeros(4, foo_parameter.n, m, n, dtype=np.complex64)

# Parallelize your for loop using joblib
total_pixels = m*n
nthreads = 8
workers_1d_idxs = np.arange(total_pixels)
workers_idxs = np.unravel_index(workers_1d_idxs, (M,N))
Parallel(n_jobs=nthreads, backend="multiprocessing", verbose=10)(delayed(reconstruct_cube)(
        F, data, sigma, nu, 0.0, workers_idxs, i, eta, False) for i in range(0, total_pixels))
```

```python
def reconstruct_cube(F=None, data=None, sigma=None, nu=None, spectral_idx=None, noise=None,
                     workers_idxs=None, idx=None, eta=1.0, use_wavelet=True):
    i = workers_idxs[0][idx]
    j = workers_idxs[1][idx]

    if spectral_idx is None:
        spectral_idx = 0.0

    dataset = Dataset(nu=nu, sigma=sigma, data=data[:, i, j], spectral_idx=spectral_idx)
    parameter = Parameter()
    parameter.calculate_cellsize(dataset=dataset, oversampling=8, verbose=False)

    dft = DFT1D(dataset=dataset, parameter=parameter)
    nufft = NUFFT1D(dataset=dataset, parameter=parameter, solve=True)

    F_dirty = dft.backward(dataset.data)

    # We can estimate the noise from the edges of the FDF
    edges_idx = np.where(np.abs(parameter.phi) > parameter.max_faraday_depth / 1.5)
    noise = eta * 0.5 * (np.std(F_dirty[edges_idx].real) + np.std(F_dirty[edges_idx].imag))

    # We store the FDF
    F[0, :, i, j] = F_dirty

    # Let's say that if use_wavelet is True then we use the coif2 wavelet
    if use_wavelet:
        wav = UndecimatedWavelet(wavelet_name="coif2")
    else:
        wav = None

    # We estimate lambda for L1 norm
    lambda_l1 = np.sqrt(2 * len(dataset.data) + np.sqrt(4 * len(dataset.data))) * noise
    chi2 = Chi2(dft_obj=nufft, wavelet=wav)
    l1 = L1(reg=lambda_l1)
    F_func = [chi2, l1]
    f_func = [chi2]
    g_func = [l1]

    F_obj = OFunction(F_func)
    g_obj = OFunction(g_func)

    parameter.data = F_dirty
    parameter.complex_data_to_real()

    if use_wavelet:
        parameter.data = wav.decompose(parameter.data)

    opt = FISTA(guess_param=parameter, F_obj=F_obj, fx=chi2, gx=g_obj, noise=noise, verbose=False)
    obj, X = opt.run()

    if use_wavelet:
        X.data = wav.reconstruct(X.data)

    X.real_data_to_complex()
    F_residual = dft.backward(dataset.residual)
    F[1, :, i, j] = X.data
    F[2, :, i, j] = X.convolve(normalized=True) + F_residual
    F[3, :, i, j] = F_residual
```

Note that if your Faraday depth cube is large, then probably it won't fit in your memory. Therefore, we can use `memory map`. In that case you would need to define your Faraday depth cube as:

```python
output_file_mmap = os.path.join(folder, 'output_mmap')
F = np.memmap(output_file_mmap, dtype=np.complex64, shape=(4, foo_parameter.n, M, N), mode='w+')
```

## Contact

Please if you have any problem, issue or you catch a bug using this software please use the [issues tab](https://github.com/miguelcarcamov/csromer/issues) if you have a common question or you look for any help please use the [discussions tab](https://github.com/miguelcarcamov/csromer/discussions).
