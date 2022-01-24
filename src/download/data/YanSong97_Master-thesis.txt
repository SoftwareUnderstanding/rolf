# Last layer Bayesian Recurrent neural network and application in reinforcement learning

This is a master final project of MSc. Computational Statisitcs and Machine Learnning program at UCL.

## Abstract:
Combining Bayesian modelling with deep learning methods has always been a research area withgreat potential.  In this project, we present a last-layer Bayesian recurrent neural network which is a simplified version of fully-Bayesian RNN experimented in many literature. It places a distribution on the last-layer parameters of the recurrent unit and offer estimate of two types of uncertainty: aleatoric and epistemic uncertainty. We have shown that such a partially-Bayesiandynamic model is sufficient to give descent prediction on a simple noise-free control environment and capture the uncertainty when the dynamic becomes stochastic.

Meanwhile, The presented model is applied in model-based reinforcement learning (MBRL) in which we have shown that the incorporation of epistemic uncertainty provides stable and robust policy learning on both noise-free and noisy tasks.  Furthermore, the model has also been shown to outperform a probabilistic dynamic model with no epistemic uncertainty estimate.



## Models implemented:

* Deterministic LSTM (DLSTM): a standard LSTM model with determinsitic transition and generating function.

* Recurrent state-space model (RSSM-LSTM): an integration of latent variable model and recurrent neural network. The transition function is determinsitic but the data uncertainty is captured.

* Last-layer Bayes LSTM (LLB-LSTM): a partially-Bayesian recurrent neural network which has stochastic weight parameters only in the last layer of the recurrent cell, implying the stochasticity in the transition function. The genrating function is also probabilistic in a way that the prediction contains a mean and a variance as well.

## Gradient-based planning:

The planning is performed on classic control task: determinisitc and stochastic cartpole balancing.

In the report we only show the reparametrisation (RP) policy gradient in which the gradient backpropagates directly through the reward function, we also implement the likelihood ratio (LR) policy gradient which is also known as REINFORCE algorithm.

## Code description:

### [Model](https://github.com/YanSong97/Master-thesis/tree/master/models):

* Deterministic RNN.py : DLSTM

* LLB RNN.py : Last layer Bayesian LSTM

* Noisy RNN.py : Based on determinstic LSTM but the generating function output the mean and variance.

* RSSM.py : Recurrent state-space LSTM

* customised LLB.py : This is a version of LLB where we overwrite the original Pytorch LSTMCell class.

### [Env](https://github.com/YanSong97/Master-thesis/tree/master/env):

* cart-pole balancing.py : Modified from gym.cartpole-v0, the label to put into 'add_noise()' function represent different degree of noise

### [Controller](https://github.com/YanSong97/Master-thesis/tree/master/controller)

* Deterministic controller.py : a determinsitic policy network for continuous action

* Discrete controller.py : a determinsitic policy network for discrete action

* Gaussian controller.py : a Gaussian policy network for continuous action

* Noisy-net Gaussian controller.py : a state-dependent exploratory policy network replicated from paper: https://arxiv.org/abs/1706.10295

### [Agent](https://github.com/YanSong97/Master-thesis/tree/master/Agent)

* Agent.py : an integrated class for environment roll-out, model training and policy learning

### [Planning](https://github.com/YanSong97/Master-thesis/tree/master/Planning)

* Learning-Planning iteration.py : main() function, iterating between model leanring and policy leanring

### [Notebook](https://github.com/YanSong97/Master-thesis/tree/master/Notebook)

This contain the demo for performing planning experiment for all three models: DLSTM, RSSM-LSTM and LLB-LSTM.

### [Report](https://github.com/YanSong97/Master-thesis/blob/master/master%20thesis.pdf)






