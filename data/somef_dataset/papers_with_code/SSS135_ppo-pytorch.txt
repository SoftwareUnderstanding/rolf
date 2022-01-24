# Proximal Policy Optimization in PyTorch

## Changes from PPO paper

####  A More General Robust Loss Function (Barron loss)
https://arxiv.org/abs/1701.03077

State-value is optimized with Barron loss. Advantages are scaled using Barron loss derivative.
To use MSE loss for state-value and unscaled advantages set `barron_alpha_c = (2, 1)`.

On average when used with Atari, instead of MSE / Huber loss, it does not change performance much. 


#### Policy / value clip constraint is multiplied by `abs(advantages)`
This will make constraint different for each element in batch.
Set `advantage_scaled_clip = False` to disable.

As with Barron loss, on average I haven't observed much difference with or without it.

#### KL Divergence penalty implementation is different
When `kl < kl_target` it is not applied.
When `kl > kl_target` it is scaled quadratically based on `abs(kl - kl_target)`
and policy and entropy maximization objectives are disabled.

I've found this implementation to be much easier to tune than original KL div penalty.

#### Additional clipping constraint
New constraint type which clips raw network output vector instead of action log probs.
See 'opt' in `PPO` `constraint` documentation.

Sometimes it helps with convergence on continuous control tasks when used with `clip` or `kl` constraints.

#### Several different constraints could be applied at same time
See `PPO` `constraint` documentation.

#### Entroy added to reward

Entropy maximization helps in some games. See `entropy_reward_scale` in `PPO`.

#### Extra Atari network architectures
In addition to original network architecture, biggger one is available. See `cnn_kind` in `CNNActor`.

#### Quasi-Recurrent Neural Networks
https://arxiv.org/abs/1611.01576

See `PPO_QRNN`, `QRNNActor`, `CNN_QRNNActor`. 
QRNN implementation requires https://github.com/salesforce/pytorch-qrnn. 
With some effort QRNN could be replaced with another RNN architecture like LSTM or GRU.

## Installation

`pip install git+https://github.com/SSS135/ppo-pytorch`

Required packages:
- PyTorch 0.4.1
- gym
- [tensorboardX](https://github.com/lanpa/tensorboardX)
- [pytorch-qrnn](https://github.com/salesforce/pytorch-qrnn) (only if using QRNN)

## Training

Training code does not print any information to console. Instead it logs various info to Tensorboard.

#### Classic control
`CartPole-v1` for 500K steps without CUDA (`--force-cuda` to enable it, won't improve performance)

`python example.py --env-name CartPole-v1 --steps 500_000 --tensorboard-path /tensorboard/output/path`

#### Atari
`PongNoFrameskip-v4` for 10M steps (40M emulator frames) with CUDA

`python example.py --atari --env-name PongNoFrameskip-v4 --steps 10_000_000 --tensorboard-path /tensorboard/output/path`


## New gym environments

When library is imported following gym environments are registered:

Continuous versions of Acrobot and CartPole `AcrobotContinuous-v1`, `CartPoleContinuous-v0`, `CartPoleContinuous-v1`

CartPole with 10000 steps limit `CartPoleContinuous-v2`, `CartPole-v2`

## Results

#### PongNoFrameskip-v4
<img src="images/pong.png" width="500">
Activations of first convolution layer
<img src="images/pong_activations.png" width="300">
Absolute value of gradients of state pixels (sort of pixel importance)
<img src="images/pong_attention.png" width="300">

#### BreakoutNoFrameskip-v4
<img src="images/breakout.png" width="500">

#### QbertNoFrameskip-v4
<img src="images/qbert.png" width="500">

#### SpaceInvadersNoFrameskip-v4
<img src="images/spaceinvaders.png" width="500">

#### SeaquestNoFrameskip-v4
<img src="images/seaquest.png" width="500">

#### CartPole-v1
<img src="images/cartpole-v1.png" width="500">

#### CartPoleContinuous-v2
<img src="images/cartpolecontinuous-v2.png" width="500">
