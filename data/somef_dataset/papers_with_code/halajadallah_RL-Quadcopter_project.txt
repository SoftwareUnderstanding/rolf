# Udacity Machine Learning nanodegree program
## Reinforcement Learning project
### Train a quadcopter to fly

The goal of this project is to train a quadcopter to perform a particular task. The task in my submission is taking off.
For this reinforcement learning project both the states and actions spaces are continuous. In such cases it efficient to apply reinforcement learning using the Deep Determinist Policy Gradients (DDPG). The DDPG methods is based on deterministic policy gradient and apply the actor-critic method. 

For details see the [DDPG paper](https://arxiv.org/abs/1509.02971):

Lillicrap, Timothy P., et al: __Continuous control with deep reinforcement learning__, ArXiv 2016.

Udacity project repository at this [link](https://github.com/udacity/RL-Quadcopter-2)

The repo has:

* agents 
* task.py: defines the task environment (I determined the reward function)
* physics_sim.py: simulates a quadcoptor
* agents folder has agent.py where I developed the agent (project requirement) 
