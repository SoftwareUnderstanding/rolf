# NF
Normalizing flows for density estimation with built-in support for sampling.

This repo provides an interface for creating normalizing flows.
It hosts two modules, `NF` and `BAM`.

The `NF` module provides a basic interface for normalizing flows, and defines some transformations like rotation, permutation, etc.
NF transformations can be stacked to create more complex NF transformations.
A NF transformation can be converted to a generative model that can be trained to model an unknown target distribution and sample from it.
See `RealNVP_test.ipynb` for an example of how the `NF` module can be used to implement the RealNVP architecture from the paper "Density estimation using Real NVP" (<https://arxiv.org/pdf/1605.08803.pdf>).

The `BAM` module provides a basic interface for blockwise autoregressive monotonic transformations, and defines some of those (most notably the `BAM.Linear`, which is the basic building block for the B-NAF architecture from the paper "Block Neural Autoregressive Flow" (<https://arxiv.org/pdf/1904.04676.pdf>)).
Individual BAM transformations can be stacked/concatenated/summed to create more complex transformations.
Similar to the `NF` module, the `BAM` module can easily be extended by custom transformations.
A BAM transformation can be converted to a NF transformation.
`B-NAF_test.py` provides an example of how the `BAM` module can be used to create a custom normalizing flow.

Python notebooks sometimes do not render on github. In this case, you can use nbviewer:
- <https://nbviewer.jupyter.org/github/sshish/NF/blob/master/RealNVP_test.ipynb>
- <https://nbviewer.jupyter.org/github/sshish/NF/blob/master/B-NAF_test.ipynb>

This repo was inspired by
- <https://github.com/senya-ashukha/real-nvp-pytorch>, which is a PyTorch implementation of RealNVP
- <https://github.com/nicola-decao/BNAF>, which is the authors' original implementation of B-NAF

Suggestions for improvement are welcome.

## Requirements
To use `NF` and `BAM` modules:
- PyTorch version 1.3.0 or above

To run the `RealNVP_test.ipynb` and `B-NAF_test.py` examples:
- CUDA support
- scikit-learn
- matplotlib
