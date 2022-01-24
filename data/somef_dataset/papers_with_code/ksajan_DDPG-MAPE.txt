# DDPG-MAPE

Reinforcement Learning MOVE37 Course Final Project

DQN 1 vs 1 | DQN 1 vs 2 | DQN 2 vs 1
:---------:|:----------:|:-----------:
![](tests/dqn_1vs1.gif "DQN 1 vs 1") | ![](tests/dqn_1vs2.gif "DQN 1 vs 2") | ![](tests/dqn_2vs1.gif "DQN 2 vs 1")


# Multi-Agent Particle Environment
This repository contains our implementation of DQN, DDPG, and MADDPG that works on a slightly modified version of the predator-pray environment. It also contains our results, including trained weights and training rewards and losses.
A simple multi-agent particle world with a continuous observation and discrete action space, along with some basic simulated physics.

Used in the paper [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/pdf/1706.02275.pdf).

# Requirements
Keras, Open AI gym, Tensorflow 

## Code structure

- `make_env.py`: contains code for importing a multiagent environment as an OpenAI Gym-like object.

- `./multiagent/environment.py`: contains code for environment simulation (interaction physics, `_step()` function, etc.)

- `./multiagent/core.py`: contains classes for various objects (Entities, Landmarks, Agents, etc.) that are used throughout the code.

- `./multiagent/rendering.py`: used for displaying agent behaviors on the screen.

- `./multiagent/policy.py`: contains code for interactive policy based on keyboard input.

- `./multiagent/scenario.py`: contains base scenario object that is extended for all scenarios.

- `./multiagent/scenarios/`: folder where various scenarios/ environments are stored. scenario code consists of several functions

## Paper citation

If you used this environment for your experiments or found it helpful, consider citing the following papers:

Environments in this repo:
<pre>
@article{lowe2017multi,
  title={Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments},
  author={Lowe, Ryan and Wu, Yi and Tamar, Aviv and Harb, Jean and Abbeel, Pieter and Mordatch, Igor},
  journal={Neural Information Processing Systems (NIPS)},
  year={2017}
}
</pre>

Original particle world environment:
<pre>
@article{mordatch2017emergence,
  title={Emergence of Grounded Compositional Language in Multi-Agent Populations},
  author={Mordatch, Igor and Abbeel, Pieter},
  journal={arXiv preprint arXiv:1703.04908},
  year={2017}
}
</pre>
