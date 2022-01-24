# Continues Control - Moving Arm
### Introduction
This project, describes the reinforcement learning to resolve the continues control problem. 
The problem is describe with continues state space and continues action space.
The goal is to hold the ball with moving arm :) 
I used TD3 algorithm which is extension of Deep Deterministic Gradient Policy method. I include my private extension of local exploration. more details in Report.pdf and https://arxiv.org/pdf/1802.09477.pdf

The enviroment comes from Unity, please read the Unity Environment, before making a copy and trying yourself!

### Get Started 
Clone the repository, install the Unity Enviroment and start with ExperienceManager.py (update UNITY_ENVIROMENT before run)

### Enviroment description
**You can see trained agent in action [here](https://www.youtube.com/watch?v=kldATbEf1zE)**

A reward of +0.1 is provided for holding the ball at each step. The maximum reward is 40 points.   
The state space has 33 dimensions and contains the position, rotation, velocity, and angular velocities of the arm. 
The action has 4 dimension in range -1 to 1. And describe the tourge to each part of arm.

The task is episodic, and in order to solve the environment, your agent must get an average *score of +30 over 100* consecutive episodes.

### Project structure
The project in writen in python and using the pytorch framework for Deep Neural network. More requirements read bellow.
The project files are following:
- ExperienceManager.py - the main file.  Responsible for run the experience, run episode, interact with agent and store statistic of episode reward 
- Agent.py - responsible for choosing action in particular state, interact with memory, and learning process             
- Memory.py - class is reponsible for storing data in array data structure, and randomly sampling data from it
- NeuralNetwork.py - define the Neural network model in Pytorch, For both Actor and Critic NN. Critic class contains definition of two NN as describe in TD3 algorithm.
- EnvironmentWrapper.py -  responsible for creating and interaction with Unity env. 
- Config.py - class hodl the all hyperparams and object required by agent
- Util.py - miscellaneous functions like, prepare model file name, store graph
- actor.pth - the learned neural network model, for interacting actions
- critic.pth - the learned neural network model provides the Q-Value function
- Report.pdf - describe the work process and some hyper parameter testing

### Installation requirement
The project was tested on 3.6 python and requires the following packages to be installed:
- numpy 1.16.4
- torch 1.1.0
- matplotlib 3.1.0
- unityagent 0.4.0

### Unity Environment
After success instalation please navigate to line num 13 in ExperienceManager.py and update the path to your installation directory
To try it yourself and see how wise you agent can be :), you'll need to download a new Unity environment.
You need only select the environment that matches your operating system:

* Linux: [download here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
* Mac OSX: [download here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
* Windows (32-bit): [download here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
* Windows (64-bit): [download here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)


