# foxi

### Futuristic Observations and their eXpected Information

<a href="http://ascl.net/1806.030"><img src="https://img.shields.io/badge/ascl-1806.030-blue.svg?colorB=262255" alt="ascl:1806.030" /></a>
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4703779.svg)](https://doi.org/10.5281/zenodo.4703779)

This python package computes a suite of expected utilities which are based on information theory and Bayesian inference, given futuristic observations, in a flexible and user-friendly way. For example, in principle, all one needs to make use of `foxi` is a set of n-dim prior samples for each model and one set of n-dim samples from the current data. 

05/08/2018 - New features including support for forecast Fisher matrices!!! 

### The expected utilities used include...

1. The expected ln-Bayes factor between models and its Maximum-Likelihood averaged equivalent (see: [Hardwick, Vennin & Wands (2018)](https://iopscience.iop.org/article/10.1088/1475-7516/2018/05/070)).

2. The decisiveness between models (see: [Hardwick, Vennin & Wands (2018)](https://iopscience.iop.org/article/10.1088/1475-7516/2018/05/070)) and its Maximum-likelihood averaged equivalent, the decisivity.

3. The expected Kullback-Leibler divergence (or information gain) of the futuristic dataset.

### Main features

Flexible inputs – usable essentially for any forecasting problem in science with suitable samples. foxi is designed for all-in-one script calculation or an initial cluster run then local machine post-processing, which should make large jobs quite manageable subject to resources. We have added features such LaTeX tables and plot making for post-data analysis visuals and convenience of presentation. In addition, we have designed some user-friendly scripts with plenty of comments to get familiar with `foxi`.

## Getting started

To fork, simply type into the terminal:

> git clone https://github.com/umbralcalc/foxi.git 

In the `/foxiscripts` directory there is an ipython notebook with a worked 5-dimensional example available [here](https://github.com/umbralcalc/foxi/tree/master/foxiscripts/5D_example.ipynb). 
This demonstrates most of the main features within the `foxi` class.



