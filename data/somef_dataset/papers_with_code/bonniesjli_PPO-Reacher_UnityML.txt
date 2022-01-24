[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"


# PPO for Reacher UnityML

This is an implementation of the Proximal Policy Optimization (Schulmann et al, 2017) for the Reacher environment in Unity ML
![Trained Agent][image1]

### Project Details

#### Environment

In the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment, where a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

#### Distributed Training

This project uses the parallel version of the Reacher environment. The version contains 20 identical agents, each with its own copy of the environment

#### Solving the environment

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30. 

I have yet to run the full implementation to solve the environment. 

### Getting Started


### Instructions
Run PPO.ipynb to run the agent. 

