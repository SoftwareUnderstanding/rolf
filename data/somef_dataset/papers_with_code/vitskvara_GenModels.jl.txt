# GenModels.jl
A collection of deep neural generative models based on the Flux library. Most models have a simple constructor for convolutional versions. For details on how to use this package, see the tests.

## Models implemented:

| acronym | name | paper |
|---------|------|-------|
| AE | Autoencoder | Vincent, Pascal, et al. "Stacked denoising autoencoders: Learning useful representations in a deep network with a local denoising criterion." Journal of Machine Learning Research 11.Dec (2010): 3371-3408. [link](http://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf)|
| VAE | Variational Autoencoder | Kingma, Diederik P., and Max Welling. "Auto-encoding variational bayes." arXiv preprint arXiv:1312.6114 (2013). [link](arxiv.org/abs/1312.6114) |
| TSVAE | Two-stage Variational Autoencoder | Dai, Bin, and David Wipf. "Diagnosing and enhancing vae models." arXiv preprint arXiv:1903.05789 (2019). [link](https://arxiv.org/abs/1903.05789) |
| AAE | Adversarial Autoencoder | Makhzani, Alireza, et al. "Adversarial autoencoders." arXiv preprint arXiv:1511.05644 (2015). [link](https://arxiv.org/abs/1511.05644) |
| WAE | Wasserstein Autoencoder with MMD loss | Tolstikhin, Ilya, et al. "Wasserstein auto-encoders." arXiv preprint arXiv:1711.01558 (2017). [link](https://arxiv.org/abs/1711.01558) |
