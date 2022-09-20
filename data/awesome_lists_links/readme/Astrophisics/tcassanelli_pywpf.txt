# PyWPF

Waterfall Principal Component Analysis (PCA) Folding.

This repository is intended to show the basic description for the Waterfall Principal Component Analysis (PCA) Folding, an alternative to find periodicity in a one dimensional timestream data sets.
Given a timestream, with each point being the arrival times of a source, the software computes the estimated period. The steps of the code follow:

1. Bin the data with a time bin, `dt`
2. Choose a trial period `T_l`
3. Divide the binned stream into equal size parts of `M` sections (number of divisions)
4. Compute the PCA for the `[M, N]` array, where `N = round(T_t / dt)`
5. Repeat the process for the number of trials `L`

Once we have run all the `L` trials we have a new set of information, which includes eigenvalues, and eigenvectors. At this stage we should define a merit function which uses these outputs to find the best trial period `T_l`. For example, taking the first eigenvalues will give a good initial estimate for the optimum. Other merit functions are listed [here](https://github.com/tcassanelli/pywpf/blob/main/pywpf/merit_functions.py).


Be aware that the software is slow and other methods, such as chi-squared epoch folding may work best. The aim of PyWPF is to find an estimated period at very high noise situations where traditional methods may fail.

Note that the software has not been optimized by any means. To speed up the process we suggest to use `mpi4py` or other parallel programing software. A faster version will be updated soon, i.e., `cython` or similar.

## Install

The only way to install the software, for now, is by cloning the source code.

```bash
git clone git@github.com:tcassanelli/pywpf.git
cd pywpf
python3 setup.py install
````
## Usage

The core function of analysis is `pywpf.pca_folding`, which requires several initial parameters in order to be run. As a ruler of thumb always use the best known period of the source (`T_init`). The solution will always be sensitive to your time binning, in particular, you should always use a time bin (`dt`) that is a couple orders smaller than the searched period.

To run a simple example code please go to link, download the data set and add the path to the script, `times_path`.
The software may be slow (not optimized!), we suggest that use it under a parallel programming script such as MPI (`mpi4py`) or something similar.

The following example runs over the Crab pulsar data (PSR B0531+21).

```python
import os
import numpy as np
import pywpcaf

dt = 0.0001                   # s
T_init = 0.0336372543236884   # s  
delta = 1e-9                  # s
iteration = 10000
base_dir = "/my/work/dir"
times_path = os.path.join(base_dir, "data_B0531+21.npy")

M = [200]                      # number of divisions, we usually iterate here
pywpf.pca_folding(
    times_path=times_path,
    dt=dt,                     # time bin
    T_init=T_init,             # initial search period
    iteration=1000,            # number of iterations
    delta=delta,               # range for search
    num_div=M,                 # number of divisions
    merit_func=pywpf.merit1,   # merit function, or create your own!        
    work_dir=base_dir
    )
```

The software will create an output directory in `work_dir` and plot multiple figures.

### Example scripts

In the [scripts/](https://github.com/tcassanelli/pywpf/tree/main/scripts) directory, not part of the package itself, we have added example scripts with real data to create some of the exciting results that PyWPF can create in high and low noise situations. Please feel free to use them!

## License

PyWPF is licensed under a 3-clause BSD style license - see the [LICENSE](https://github.com/tcassanelli/pywpf/blob/main/LICENSE.md) for more information.

## TODOs

- [ ] Cythonize some sections of the code
- [ ] Include a test suite 
- [ ] Documentation
- [ ] Add `astropy` compatibility, with units and other functions
- [ ] Include native chi-squared epoch folding for fast computations
- [X] Include a license for the code

## Acknowledgements
 
We have submitted a paper to Astronomy & Astrophysics. We'll update here how to cite our work. Thanks!
