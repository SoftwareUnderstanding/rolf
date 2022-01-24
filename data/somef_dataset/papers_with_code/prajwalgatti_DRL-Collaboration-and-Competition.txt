
[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"

## Description

![Trained Agent][image1]

The goal of the project is to create an agent that learns how to efficiently solve a Tennis environment made with Unity-ML agents. While active the agent is trying to approximate the policy that defines his behaviour and tries to maximize the performance in the context of the environment.

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

## Learning Algorithm Implemented


The reinforcement learning agent implementation follows the ideas of [arXiv:1509.02971](https://arxiv.org/abs/1509.02971) paper implementing a DDPG agent.

DDPG is an actor-critic method.

The actor network is responsible for chosing actions based on the state and the critic network try to estimate the reward for the given state-action pair.

DDPG in continuous control is particularly useful as, unlike discrete actions, all the actions are "choose" at every timestep with a continuous value making non-trivial to build a loss function based on these values.

Instead, the actor network is indirectly trained using gradient ascent on the critic network, reducing the problem of building a loss function to a more classic RL problem of maximize the expected reward.

The agent exploits the initial lack of knowledge as well as [Ornstein–Uhlenbeck process](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process) -generated noise to explore the environment.

The algorithm also leverages the fixed-Q target, double network, soft-updates and experience replay.

The hyperparameters selected for the demonstration are:

- Actor learning rate: 0.0001
- Critic learning rate: 0.0001
- Update rate: 1
- Memory size: 100000
- Batch size: 128
- Gamma: 0.99
- Tau: 0.001
- Adam weight decay: 0
- Number of episodes: 9000

Actor & Critic networks:
```
Actor(
  (fc1): Linear(in_features=24, out_features=256, bias=True)
  (fc2): Linear(in_features=256, out_features=128, bias=True)
  (fc3): Linear(in_features=128, out_features=2, bias=True)
)
```
```
Critic(
  (fcs1): Linear(in_features=24, out_features=256, bias=True)
  (fc2): Linear(in_features=258, out_features=128, bias=True)
  (fc3): Linear(in_features=128, out_features=1, bias=True)
)
```


## Plot of Rewards 

![](https://github.com/prajwalgatti/DRL-Collaboration-and-Competition/blob/master/plot.png)

The saved weights of the Actor and Critic networks can be found [here.](https://github.com/prajwalgatti/DRL-Collaboration-and-Competition/tree/master/savedmodels)

It took the networks 2087 episodes to be able to perform with not less than score of 0.500 as an average of 100 episodes.
Training it different times give us more or less number of episodes to solve sometimes. It can also be reduced by tuning the hyperparameters.

Follow the setup [here.](https://github.com/prajwalgatti/DRL-Collaboration-and-Competition/blob/master/Setup_instructions.md)

Follow this [notebook](https://github.com/prajwalgatti/DRL-Collaboration-and-Competition/blob/master/Tennis.ipynb) for training models.


## Ideas for Future Work

- Perform search for better hyperparameters of algorithm as well as the neural networks
- Implement a state to state predictor to improve the explorative capabilities of the agent
