# VQ-VAE Implementation in PyTorch
This repo is an implementation of [VQ-VAE](https://papers.nips.cc/paper/2017/file/7a98af17e63a0ac09ce2e96d03992fbc-Paper.pdf) using PyTorch
This implementation is not official, this repo is just for study!

# Experiments
All code about training and inference is in "train_vqvae-*.ipynb" files.
I trained VQ-VAE with [STL-10](https://cs.stanford.edu/~acoates/stl10/) dataset.

Following pictures are result of experiments. All pictures are test image.
* VQ loss version  
![Image](resources/vqvae.png)
* EMA version  
![Image](resources/vqvae-ema.png)


# Reference
* Oord, A., Oriol Vinyals and K. Kavukcuoglu. “Neural Discrete Representation Learning.” NIPS (2017).
* Adam Coates, Honglak Lee, Andrew Y. Ng An Analysis of Single Layer Networks in Unsupervised Feature Learning AISTATS, 2011.