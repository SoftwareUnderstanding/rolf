# VAE

This repo provides a basic pytorch implementation of the Variational Autoencoder (VAE) as described in https://arxiv.org/pdf/1312.6114.pdf.

The implementation is specific to the binarized MNIST dataset, even if it can be easily adapted for other kinds of data.

Note also that – again, in this implementation – both `p(z)` and `q(z|x)` are Gaussian. Thus, as shown in the paper, the `KL divergence` in the loss has a closed-form expression.

## Training and generating images

To train the model and generate images, run 

```
python main.py
```

One may also pass some additional parameters through the command line. For instance, run

```
python main.py -D 10
```

to set the dimensionality of the latent space equal to `10`.

For the full list of the command line arguments, see the script `args.py`.

## 2-D learned manifold

Additionally, if `D=2`, the learned data manifold will be produced as well. Here is an example of it:

![manifold](images/manifold_hl:1_hu:500_D:2_b:128_L:10_e:20.png)
