# jax-dcgan

<a href="https://colab.research.google.com/github/bkkaggle/jax-dcgan/blob/main/dcgan.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

A simple implementation of the [DCGAN](https://arxiv.org/abs/1511.06434) paper trained on the MNIST dataset using Jax and Flax.

The Jupyter notebook provided can be opened directly to Google Colab and trains on the provided TPU and generates decent samples in less than two minutes.

This is a sort-of toy example to show how to implement a GAN in Flax so I stuck to using a small dataset (MNIST) so that the notebook doesn't take too long to train. I've limited the notebook to train the GAN for 2000 iterations and with a relatively small batch size of 32 per core (32 imgs/core * 8 cores = 256 imgs/batch), so you'd likely have to increase these when training on a larger dataset or with a larger model.

If you're interested in training or using DCGAN in another ml frameworks, check out the official [PyTorch tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html) and the official [Tensorflow tutorial](https://www.tensorflow.org/tutorials/generative/dcgan)
