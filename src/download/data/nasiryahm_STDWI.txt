# STDWI

Implementation of a spike timing-dependent weight Inference (STDWI) method and competitors -- all of which are proposed as biologically plausible methods to solve the weight transport problem for the backpropagation of error algorithm.

In this repository we have our implementation of the STDWI method, the regression discontinuity design (RDD) method by Guerguiev et al. and a modified rate-based method by Akrout et al.
See [Example.ipynb](./Example.ipynb) for a walkthrough of simulating a feedforward network of leaky integrate and fire neurons and inference of the synaptic weights using these techniques.

The scripts used to produce plots shown in our [arXiv pre-print](http://arxiv.org/abs/2003.03988) are located in the [paper_scripts](./paper_scripts/) folder.

This repository contains a number of useful files and scripts:

- `./weight_inference/` This folder contains a python library which contains all fundamental functions and code used to produce the relevant results
- `./Example.ipynb` This file provides an example of loading the library enclosed and shows a single comparative example of weight inference
- `./paper_scripts` This folder contains (in a set of sub-directories) the key python scripts which were executed to produce the data and plots for our submitted paper. Note that each of these scripts should be executed with their directory as the current path. Note also that for multiple repeats (seeds) or for parameter searches, these files require some modification (see comments). 
- `./conda_requirements.txt` This file describes the specific (conda-based) python environment and all packages which were installed and leveraged to carry out simulations.



Guerguiev, J., Kording, K. P., & Richards, B. A. (2019). Spike-based causal inference for weight alignment. In arXiv [q-bio.NC]. arXiv. http://arxiv.org/abs/1910.01689

Akrout, M., Wilson, C., Humphreys, P. C., Lillicrap, T., & Tweed, D. (2019). Deep Learning without Weight Transport. In arXiv [cs.LG]. arXiv. http://arxiv.org/abs/1904.05391

