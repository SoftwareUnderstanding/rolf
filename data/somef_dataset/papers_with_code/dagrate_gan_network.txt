# gan_network

The gan_network library is a Python library that proposes Wasserstein Generative Adversarial Network (WGAN) for numerical simulations. Two types of WGAN are implemented: 
- a standard WGAN as described in [1] and [2]
- a gradient penalty GAN as described in [3]

----------------------------

## Dependencies

The library uses **Python 3** and the following modules:
- numpy (pip install numpy)
- keras (pip install numba)
- pylab (pip install pylab)
- functools (pip install pylab)
- sklearn (pip install sklearn)

If running in Linux distribution, it is advised to install BLAS/LAPACK to increase the efficiency of the computations:  
sudo apt-get install libblas-dev liblapack-dev gfortran

----------------------------

## Quick Start

The standard WGAN is available in the file WGAN_std.py

The gradient penalty WGAN is available in the file WGAN_gradientpenalty.py

```python
```

----------------------------

## References

[1] Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. 2014. Generative adversarial nets. In Proceedings of the 27th International Conference on Neural Information Processing Systems - Volume 2 (NIPS'14), Vol. 2. MIT Press, Cambridge, MA, USA, 2672-2680.

[2] Martin Arjovsky, Soumith Chintala, Léon Bottou. 2017, Wasserstein Generative Adversarial Networks. Proceedings of the 34th International Conference on Machine Learning, PMLR 70:214-223, 2017. Paper available at http://proceedings.mlr.press/v70/arjovsky17a/arjovsky17a.pdf

[3] Gulrajani, Ishaan & Ahmed, Faruk & Arjovsky, Martin & Dumoulin, Vincent & Courville, Aaron. (2017). Improved Training of Wasserstein GANs. Paper available at https://arxiv.org/pdf/1704.00028.pdf

----------------------------

## Citing 

Further to come
