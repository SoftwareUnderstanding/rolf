# hibayes

Framework for fully-bayesian Monte-Carlo analysis of the global HI
spectrum from the Cosmic Dawn.

As written the code uses
[(py)multinest](http://ccpforge.cse.rl.ac.uk/gf/project/multinest) for
sampling in order to obtain the evidence as well as the posterior, but
feel free to plug in your own sampler, MCMC or otherwise.

## Overview

- **Framework** Uses a fully-bayesian framework to fit a model to the
  measured global HI spectrum. 'Fully bayesian' here means parameter
  estimation **and** model selection via the bayesian evidence
  (Occam's razor quantified). See
  e.g. [Mackay 2003](http://www.inference.phy.cam.ac.uk/mackay/itila/book.html).
- **Models** Currently implemented models are polynomial foregrounds
  and a (empirical) gaussian HI decrement, but any other parametric
  model can be coded-in straightforwardly (e.g. simulated/physical
  parameters).
- **Uncertainties** The (gaussian) form of the likelihood function
  assumes gaussian uncertainties on the input data, those
  uncertainties being propagated automatically by the sampling
  process.
- **Samplers** For sampling we have used ```multinest```, though any
  nested or MCMC sampler could be used. ```multinest``` has the
  advantage of offering the evidence as well as the posterior samples
  (a higher density of samples corresponding to a higher
  probability). Calculating the evidence is intrinsically expensive
  compared to 'vanilla' MCMC, but is perfectly doable for problems ---
  such as this one --- that have a small (< 30, say) number of
  parameters.
- **Output** The output for a given model is the bayesian evidence
  (plus uncertainty) and a 'chain' of posterior samples.
- **Modus operandi** The *modus operandi* is to use the evidence to
  select the winning model from a field of single-multinest-run
  competitors, then to examine the corresponding triangle plot (the
  'final answer') and to derive reconstructed and residual spectra.
- **Priors** Note also that there is an inescapable choice of priors
  on parameters (and models), but the evidence quantifies this.
- **Runtime** The code is in python plus MPI and takes a few minutes
  to run on a laptop, runtime depending mainly on the model complexity
  (i.e. number of parameters).
- **What hibayes isn't** Note that the algorithm is **not** maximum
  likelihood or an expensive least squares (though the likelihood
  function could be used for either of these). Rather, the sampler is
  used to explore the full posterior probability distribution of the
  model parameters and so unmask any degeneracies, multimodalities,
  correlations, wings, skirts, etc. of/between parameters.
- **Joint analysis** It is easy to add in the future a joint
  likelihood over multiple data sets/experiments or incorporate models
  for telescope systematics.

## Software requirements

- python 2.7.x
- MPI (optional)
- mpi4py
- [multinest](http://ccpforge.cse.rl.ac.uk/gf/project/multinest)
- [pymultinest](http://johannesbuchner.github.io/PyMultiNest)

## Install

1. Fetch and install multinest, enabling MPI for an optional ~ ```NPROCS```
speed-up (see below). Check the multinest examples run OK.

2. Install mpi4py and pymultinest:

```pip install mpi4py```

```pip install pymultinest```

Don't forget to set the ```(DY)LD_LIBRARY_PATH``` environment (see
[here](http://johannesbuchner.github.io/PyMultiNest/install.html#running-some-code)). No output from the following command indicates success:

```python -c 'import pymultinest'```

Then check the pymultinest examples run OK.

3. Don't forget to

```chmod +x *py```


## Usage

**From the project root directory,**

```mpiexec -n NPROCS ./hi_multinest.py examples/1_simulated_data/config.ini```

where ```NPROCS``` is the number of cores you wish to use (execution with
MPI typically takes a few minutes on a laptop). Without MPI, just run

```./hi_multinest.py examples/1_simulated_data/config.ini```

The ```multinest``` output goes to ```examples/1_simulated_data/output/```,
roughly as follows (see ```multinest``` README for more info; the
outstem ```1-``` can be set in ```config.ini```):

- ```1-stats.dat``` is written out every ```updInt``` iterations; the
  top line contains the evidence plus uncertainty, with the (optional)
  importance-nested-sampling (INS) evidence on line 2. Mean, ML and
  MAP parameter estimates follow. Caution --- be wary of
  overinterpreting these averages and point estimates without fully
  eyeballing the posteriors!
- ```1-post_equal_weights.dat```, populated once the multinest run is
  completed, contains the equally-weighted posterior samples, the last
  column being the value of the posterior for each sample. This file
  is used for plotting and reconstruction.
- ```1-summary.txt``` is used for reconstruction.
- ```1-ev.dat``` and ```1-phys_live.points``` are written out as
  sampling proceeds.
- ```1-IS.*``` are the corresponding files for INS.
- ```1-resume.txt``` for checkpointing.
- ```1-.txt``` for analysis in [cosmomc](http://cosmologist.info/cosmomc).

**Now create a triangle plot (PDF):**

```./hi_plot.py examples/1_simulated_data/config.ini```

Alternatively use e.g. [corner](https://github.com/dfm/corner.py):

```pip install corner```

**And generate a text file ```recon_stats.txt``` containing a
MAP-centred reconstruction with error bars (see code for details):**

```./hi_recon.py examples/1_simulated_data/config.ini```

The file ```recon_raw.txt``` contains the distributions of
reconstructed spectra and has dimensions ```nsamp x nfreqs```.

## Supplied examples (see Bernardi et al. 2016)

1. Simulation

2. LEDA data (not yet available)

3...

## Citations

Use of this code should be cited as Zwart et al. 2016 (ASCL):

```
@MISC{2016ascl.soft06004Z,
   author = {{Zwart}, J.~T.~L. and {Price}, D. and {Bernardi}, G.},
    title = "{HIBAYES: Global 21-cm Bayesian Monte-Carlo Model Fitting}",
 keywords = {Software },
howpublished = {Astrophysics Source Code Library},
     year = 2016,
    month = jun,
archivePrefix = "ascl",
   eprint = {1606.004},
   adsurl = {http://adsabs.harvard.edu/abs/2016ascl.soft06004Z},
  adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

The algorithm, applied to both simulated and LEDA data, is described
in Bernardi et al. 2016 (MNRAS accepted):

```
@ARTICLE{2016arXiv160606006B,
   author = {{Bernardi}, G. and {Zwart}, J.~T.~L. and {Price}, D. and {Greenhill}, L.~J. and 
   {Mesinger}, A. and {Dowell}, J. and {Eftekhari}, T. and {Ellingson}, S.~W. and 
   {Kocz}, J. and {Schinzel}, F.},
    title = "{Bayesian constraints on the global 21-cm signal from the Cosmic Dawn}",
  journal = {ArXiv e-prints},
archivePrefix = "arXiv",
   eprint = {1606.06006},
 keywords = {Astrophysics - Cosmology and Nongalactic Astrophysics},
     year = 2016,
    month = jun,
   adsurl = {http://adsabs.harvard.edu/abs/2016arXiv160606006B},
  adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```