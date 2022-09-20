# Planet-Finder
This repository holds the data and analysis needed for my paper that finds a low-mass companion to $\psi^1$ Draconis A. The radial velocity data is available in data/rv_data.txt, and the MCMC chains are available in data/SB2_samples.npy.  The chains are in the numpy format, so can be read in with

```python
samples = np.load('data/SB2_samples.npy')
```

The chains are available in text format upon request (they are too big for github in that format).
