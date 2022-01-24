# VAE
Implementation of the variational autoencoder (VAE) detailed in Auto-Encoding Variational Bayes (Kingma and Welling 2014, https://arxiv.org/pdf/1312.6114.pdf) for MNIST. To run:
```
python vae.py [CONFIG FILE]
```
The script will train a VAE where both the encoder and decoder are neural networks with a single hidden layer each. 

The following are the parameters in the config file:

* `dataset`: name of the dataset. Currently, only `'mnist'` is implemented
* `input_dims`: dimensions of the input, written as a list, i.e. `[28, 28]`
* `hidden_size`: size of the hidden layer
* `latent_size`: size of the latent space
* `epochs`: how many epochs to train for
* `save_freq`: the script will save a set of sample outputs and a checkpoint of the model every `save_freq` epochs
* `device`: `'cuda'` or `'cpu'`, depending if you want to train on GPU or not

Config file for MNIST is in `configs/mnist.yaml`. 