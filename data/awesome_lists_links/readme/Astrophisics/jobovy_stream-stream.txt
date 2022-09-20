# stream-stream

This repository contains the code to do the analysis in [Bovy
(2015)](http://arxiv.org/abs/1512.00452) of the interaction between a
stellar stream and a disrupting dark-matter halo.

## AUTHOR

Jo Bovy - bovy at astro dot utoronto dot ca

## Requirements

* [galpy](https://github.com/jobovy/galpy)
* [NEMO](http://bima.astro.umd.edu/nemo/); see [jobovy/nemo](https://github.com/jobovy/nemo) for a perhaps easier to install version

and the usual scientific Python packages (Numpy, Scipy, matplotlib,
seaborn).

## Code overview

### [StreamKicks.ipynb](py/StreamKicks.ipynb)

Brief notebook that computes the kicks due to the interaction of a
stellar stream with a dark-matter stream in the impulse approximation.

### [SnapshotAnalysis.ipynb](py/SnapshotAnalysis.ipynb)

(better on
[nbviewer](http://nbviewer.ipython.org/github/jobovy/stream-stream/blob/master/py/SnapshotAnalysis.ipynb), where you can toggle the code)

Notebook analyzing *N*-body simulations of the interaction between a
stellar stream and various dark-matter streams. The initial conditions
for the *N*-body simulation are computed in [this
notebook](py/Orbits-for-Nbody.ipynb).

The *N*-body simulations are run using gyrfalcON and
[NEMO](http://bima.astro.umd.edu/nemo/) and use a variety of NEMO
tools. The initial conditions and the necessary commands to run the
*N*-body simulations are given in [the sim directory](sim/). Using
these it should be straightforward to repeat the analysis in the
paper.

### [SnapshotAnalysisNFW.ipynb](py/SnapshotAnalysisNFW.ipynb)

(better on
[nbviewer](http://nbviewer.ipython.org/github/jobovy/stream-stream/blob/master/py/SnapshotAnalysisNFW.ipynb), where you can toggle the code)

Same as above, but for simulations where the DM halo is modeled as an
NFW halo rather than a Plummer sphere. The resulting tidal tails and
their effect on the globular-cluster stream are very similar for these
two cases.
