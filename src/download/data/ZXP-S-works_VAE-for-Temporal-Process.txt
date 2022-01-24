# VAE for Tenporal Sequence
Variational AutoEncoder used to built a forward model that can predicting multiple frames of a system.


## Dependencies
- PyTorch
- OpenAI gym
- numpy

## Neural Networks Structure
The structure of the VAETP neural networks is show in figure. The encoder consist of there convernutional layers, they have 16, 32, and 64 channels, respectively. The decoder has a mirrored structure, i.e., 64, 32, and 16 channels. More over, in VAETP, two single fully connected layer neural networks are added. One is the transition model, designed for predicting the \mu_\text{T} according to input z_{t+1}, z_{t+2},...,z_{t+h-1}. The other is the transition sigma, designed for estimating the \Sigma_\text{T} with input \mathbf{1} \in \mathbb{R}^{dim}.

During training, the forward propagate of the neural networks will be the green array showed in figure.\ref{fig:VAETP_structure}. Note that the same encoder and the same decoder are used for each x_i multiple times, instead of one encoder or decoder for multiple images. In this way, the parameter of the networks can be reduced.

During testing, the forward propagate is labeled by the yellow array. We do not use (\mu_\text{T},\ \Sigma_{\text{T}}) to sample z_\text{T} due to noise (\Sigma_{\text{T}} is to large relative to \mu_\text{T}).

![alt text](https://github.com/ZXP-S-works/VAE-for-Temporal-Process/blob/master/VAE_CartPole.3.0/VAETP_structure.png)

## Results
The parameters are similar to the previous section. This VAETP use 5 frames to predict the next 5 frames. After 50 epochs (146 min) training, the VAETP achieved the error of transition z_\text{T} for all frames 0.04 and for the first frame 0.006. Note that this error in the previous section for the first frame is 0.03. The reconstructed images are shown in the figure.
![alt text](https://github.com/ZXP-S-works/VAE-for-Temporal-Process/blob/master/VAE_CartPole.4.3/Difference_recon_predic.png)

## Reference
1. Auto-Encoding Variational Bayes. Diederik P Kingma, Max Welling (paper): 
https://arxiv.org/abs/1312.6114
2. Reinforcement learning (dqn) tutorial
3. Tutorial - what is a variational autoencoder? – jaan altosaar
4. Lyeoni. lyeoni/pytorch-mnist-vae, Oct 2018

