# The mass distribution of Galactic double neutron stars
We highly recommend reading [*The Mass Distribution of Galactic Double Neutron Stars*; (Farrow, Zhu, & Thrane 2019)](https://iopscience.iop.org/article/10.3847/1538-4357/ab12e3/) for details along with this demonstraion.

Here we provide code which performs Bayesian inference on a sample of 17 Galactic double neutron stars (DNS) in order to investigate their mass distribution. Each DNS is comprised of two neutron stars (NS), a recycled NS and a non-recycled (slow) NS. We compare two hypotheses: A - recycled NS and non-recycled NS follow an identical mass distribution, and B - they are drawn from two distinct populations. Within each hypothesis we also explore three possible functional models: gaussian, two-gaussian (mixture model), and uniform mass distributions.

You can take a look at the [demo here](https://github.com/NicholasFarrow/GalacticDNSMass/blob/master/inferenceDemo.ipynb) or you can download the git repository with:

`git clone https://github.com/NicholasFarrow/GalacticDNSMass`.

![binary mass pdfs](demoFiles/fig_pcSamples.png)

## Requirements 
### Without running inference (just demonstration & data analysis):
* Jupyter or Ipython
* numpy, scipy

### Additional requirements if performing own inference:
* PyMultiNest (see https://johannesbuchner.github.io/PyMultiNest/install.html)

## Full code
A more detailed version of the code can be found here under [mainCode](/mainCode/).

## Citations
Thank you [Buchner et al. 2014, A&A](http://www.aanda.org/articles/aa/abs/2014/04/aa22971-13/aa22971-13.html) for their python interface of MultiNest [F. Feroz, M.P. Hobson, M. Bridges. 2008](https://arxiv.org/abs/0809.3437)
