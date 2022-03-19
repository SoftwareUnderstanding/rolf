## Description

![](https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent")

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

This projects implements DDPG for continous control for the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

## Learning Algorithm Used

The reinforcement learning agent implementation follows the ideas of [arXiv:1509.02971](https://arxiv.org/abs/1509.02971) paper implementing a DDPG agent. It is an Actor-Critic method.
The algorithm helps the agent to act in an environment with a goal of solving the task defined by the environment as well as explore the environment in order to improve the agent's behaviour. The algorithm is also augmented with the fixed-Q target, double network, soft-updates and experience replay.
The agent exploits the initial lack of knowledge as well as Ornstein–Uhlenbeck process-generated noise to explore the environment.


The hyperparameters selected for the demonstration are:

- Actor learning rate: 0.0001
- Critic learning rate: 0.0001
- Update rate: 1
- Memory size: 100000
- Batch size: 64
- Gamma: 0.99
- Tau: 0.001
- Adam weight decay: 0

- Number of episodes: 200

It took the network 114 episodes to be able to perform with not less than score of 30 as an average of 100 episodes.
Training it different times give us lesser number of episodes to solve sometimes. 
It can also be reduced by tuning the hyperparameters.

## Plot of Rewards 

![](https://github.com/prajwalgatti/DRL-Continuous-Control/blob/master/plot.png)

The saved weights of the Actor and Critic networks can be found [here.](https://github.com/prajwalgatti/DRL-Continuous-Control/tree/master/savedmodels)

Follow setup [here.](https://github.com/prajwalgatti/DRL-Continuous-Control/blob/master/Setup.md)

Train the network [here.](https://github.com/prajwalgatti/DRL-Continuous-Control/blob/master/Continuous_Control.ipynb)

## Ideas for Future Work

- Search for better hyperparameters of algorithm as well as neural network
- Implement a state to state predictor to improve the explorative capabilities of the agent
