# Decoder-based generative models

This repository contains the implementation of the following models:

- Wassertein GAN: Gulrajani, Ishaan, Faruk Ahmed, Martin Arjovsky, Vincent
  Dumoulin, and Aaron Courville. "Improved Training of Wasserstein GANs."
  Advances in Neural Information Processing Systems, March 31, 2017, 5768–78.
  http://arxiv.org/abs/1704.00028.
- Autoencoder with support to multiple feature types. Autoencoder loss may
  be masked to avoid applying gradient-based optimisation for specific features
  per example. One possible application is to avoid training model to replicate
  imputed data.

# License and warranty

© 2020 WTFPL - Werner Spolidoro Freund

This work is free. It comes without any warranty, to the
extent permitted by applicable law. You can redistribute it 
and/or modify it under the terms of the WTFPL, Version 2, 
as published by Sam Hocevar. See http://www.wtfpl.net/ 
for more details.
