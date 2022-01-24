# Duel Double DQN implementation 

This project is my sample implementation of Duel Double DQN algorithm described in detail:
https://arxiv.org/pdf/1511.06581.pdf

The environment used to present the algorithm is Banas from Unity-ML
You don’t have to build the environment yourself the prebuilt one included in the project will work fine - please note it’s only compatible with Unity-ML 0.4.0b NOT the current newest version. I don’t have access to the source of the environment as it was prebuilt by Udacity. 

## Environment details:
A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions.   

Four discrete actions are available, corresponding to:  

* 0 - move forward.  
* 1 - move backward.  
* 2 - turn left.  
* 3 - turn right.  

The problem is considered solved when the agent achieves average score of at least 13 over 100 episodes. 

## Video of the trained agent:
[![Click to watch on youtube](https://img.youtube.com/vi/SRBDl_yjLBM/0.jpg)](https://youtu.be/SRBDl_yjLBM)

## Installation: 
Please run pip install . in order to ensure you got all dependencies needed

To start up the project:
python -m train.py 

All hyper-paramters are in: 
config.py 

It includes PLAY_ONLY argument which decides whether to start Agent with pre-trained weights or spend a few hours and train it from scratch :) 

More details on the project can be found in:  
[Report](/Report.md)



