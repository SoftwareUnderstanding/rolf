# PyTorch Reimplementation - MisGAN: Learning From Incomplete Data With Generative Adversarial Networks
> MSc Coursework Project in COMP6248 Deep Learning

This project reimplements MisGAN in PyTorch according to the description in the original paper. In particular, MisGAN has 2 types of architecture: convolutional (Conv-MisGAN) and fully-connected (FC-MisGAN). The reimplementation focuses on MNIST data only for a qualitative comparison between our results and original authors'. See the [report](report.pdf) for the reimplementation detail, results, and evaluation. 

![ConvMisGAN](test/epoch300.png "Conv-MisGAN")

## Running
Source code is located in the `src` directory. Jupyter notebooks in the `test` directory can also be run in isolation.

In the `src` directory,

Conv-MisGAN on MNIST:
```
python conv_misgan.py
```

FC-MisGAN on MNIST:
```
python fc_misgan.py
```

### Requirements
This code was tested on:

- Python 3.6
- PyTorch 1.5.0
- Google Colab

## References
Research papers included in the `references` folder
- [Original Paper](https://openreview.net/forum?id=S1lDV3RcKm) (OpenReview)
- [DCGAN](https://arxiv.org/abs/1511.06434) (arXiv)
- [WGAN with Gradient Penalty](https://arxiv.org/abs/1704.00028) (arXiv)
