# What's this
Implementation of some autoencoder models in PyTorch.

I wrote this packages just for personal usage(=RL research),
but maybe it's reusable for other usages.

# Models
- AutoEncoder
  - [FC model](./pytorch_autoencoders/models/ae.py)
  - [Example](./examples/ae_mnist.py)
- Variational Auto Encoder
  - introduced in https://arxiv.org/abs/1312.6114
  - [FC model](./pytorch_autoencoders/models/vae.py)
  - [CNN model](./pytorch_autoencoders/models/conv_vae.py)
  - [Example](./examples/vae_mnist.py)
- β-VAE
  - introduced in https://openreview.net/forum?id=Sy2fzU9gl
  - model is same as VAE, only loss function is different
  - [Loss function](./pytorch_autoencoders/models/beta_vae.py)
  - [Example](./examples/betavae_dsprites.py)
- β-VAE (Burgess et al. version)
  - introduced in https://arxiv.org/abs/1804.03599
  - to avoid confusion, it's called γ-VAE in this repo.
  - [Loss function](./pytorch_autoencoders/models/gamma_vae.py)
  - [Example](./examples/gammavae_dsprites.py)

# License
This project is licensed under Apache License, Version 2.0
([LICENSE](LICENSE) or http://www.apache.org/licenses/LICENSE-2.0).


