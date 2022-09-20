# SPAM

`spam` is a python-3 package to search for imprints of Hu-Sawicki f(R) gravity on the rotation curves of the [SPARC](http://astroweb.cwru.edu/SPARC/) sample, using the MCMC sampler [emcee](http://dfm.io/emcee/current/).

This code was used to generate the results in Naik et al., (2019). Please direct any comments/questions to the author, Aneesh Naik, at an485@[Cambridge University].


## Prerequisites

This code was written and implemented with python (version 3.6.8), and requires the following external packages (the version numbers in parentheses indicate the versions employed at time of writing, not particular dependencies):

* `emcee` (2.2.1)
* `numpy` (1.16.1)
* `scipy` (1.2.0)

## Usage

Running the MCMC sampling for any of the preset models listed in Table 1 of the paper is straightforward. The following examples demonstrates how to do this for one galaxy under 'Model B'.

```python
import spam

# name of chosen galaxy
name = 'NGC2403'

# load SPARC data for galaxy
galaxy = spam.data.SPARCGalaxy('NGC2403')

# set up model B MCMC; 30 walkers and 4 temperatures
f = spam.fit.GalaxyFit(galaxy, nwalkers=30, ntemps=4, model='B')

# run 1000 iterations
f.iterate(niter=1000)
```

Note that the models including f(R) gravity can be rather time-expensive due to the computational cost of the scalar field solver (see discussion in paper). It is recommended to run these models on multiple CPU threads. See GalaxyFit documentation for instructions on how to do this.

One can inspect the MCMC chains via the `GalaxyFit.chain` attribute. The `GalaxyFit.theta_dict` attribute is a useful dictionary which translates names of parameters to indices.

Here is some code to generate a histogram showing the marginal posterior distribution for fR0 from the MCMC sampling we performed above.

```python
import matplotlib.pyplot as plt

# get the fR0 chain
# first index is 0 to get the zeroth temperature chain
# the third index slices from 500 to 'burn in' the first 500 iterations.
chain = f.chain[0, :, 500:, f.theta_dict['fR0']]

# flatten the chain
flatchain = chain.flatten()

# plot a histogram of the chain with 200 bins
plt.hist(flatchain, bins=200)
plt.show()
```

All of the `spam` output data analysed in the paper can be found [here](https://www.ast.cam.ac.uk/~an485/SPAM_fits_public/). Note that the chains, and therefore file sizes, are rather large. However, summary data for each model can be found in the 'summaries' folder. These are stored as pickled `spam.analysis.FitSummary` objects.

The scripts used to generate all of the figures in the paper can be found in the submodule `spam.plot`. These scripts search for an environment variable SPAMFITDIR, which is the directory containing all of the fit data linked above. All figures except 5) and A1) can be generated with the summary files only.

## Authors

This code was written by **Aneesh Naik** ([website](https://www.ast.cam.ac.uk/~an485/)). The research was performed in collaboration with the co-authors of Naik et al. (2019):

* [Ewald Puchwein](https://www.aip.de/Members/epuchwein)
* [Anne-Christine Davis](http://www.damtp.cam.ac.uk/user/acd/)
* [Debora Sijacki](https://www.ast.cam.ac.uk/people/Debora.Sijacki)
* [Harry Desmond](https://www2.physics.ox.ac.uk/contacts/people/desmond)


## License

Copyright (2019) Aneesh Naik and co-authors.

SPAM is free software made available under the MIT license. For details see LICENSE.

If you make use of SPAM in your work, please cite our paper ([arXiv](), [ADS]()).


## Acknowledgments

Please see the acknowledgments in the paper for a list of the many people and institutions to whom we are indebted!
