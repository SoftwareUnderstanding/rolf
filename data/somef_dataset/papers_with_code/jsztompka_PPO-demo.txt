# PPO-demo
Repository uses Unity-ML Reacher as environment for Proximal Policy Optimization agent 

This project is my sample implementation of Proximal Policy Optimization algorithm described in detail:
https://arxiv.org/abs/1707.06347

The environment used to present the algorithm is Reacher (20 arms) from Unity-ML
You don’t have to build the environment yourself the prebuilt one included in the project will work fine - please note it’s only compatible with Unity-ML 0.4.0b NOT the current newest version. I don’t have access to the source of the environment as it was prebuilt by Udacity. 

## Environment details:
A reward of +0.1 is provided for touching a ball, and a reward of 0 is returned in all other cases. Thus, the goal of the agent is to keep touching the moving ball for as long as possible. 

The state space has 33 dimensions:
* Vector Observation space: 33 variables corresponding to position, rotation, velocity, and angular velocities of the two arm Rigidbodies.
* Vector Action space: (Continuous) Size of 4, corresponding to torque applicable to two joints.
* Visual Observations: None.

Four continunous actions are available, corresponding to arm torque and angle. 

The problem is considered solved when the agent achieves average score of at least 30 over 100 episodes. 

## Installation: 
Please run pip install . in order to ensure you got all dependencies needed

To start up the project:
python -m train.py 

All hyper-paramters are in: 
config.py 

The config includes PLAY_ONLY argument which decides whether to start Agent with pre-trained weights or spend a few hours and train it from scratch :) 

More details on the project can be found in:  
[Report](/Report.md)




