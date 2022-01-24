
# DRLND
Repository related to projects for the Udacity Deep Reinforcement Learning Nano Degree
Introduction

# Introduction

For this project, we will train an agent to navigate (and collect bananas!) in a large, square world.

![Trained Agent](collect-banana.gif?raw=true "Trained Agent")

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. 
Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects 
around agent's forward direction. Given this information, the agent has to learn how to best select actions. 

Four discrete actions are available, corresponding to:

- 0 - move forward.
- 1 - move backward.
- 2 - turn left.
- 3 - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

# Prequisites

- install anaconda
- install pytorch 0.4
- install jupyter notebook (if you installed anaconda, this step is not necessary)

# Install Unity Machine Learning Agents

Unity Machine Learning Agents (ML-Agents) is an open-source Unity plugin that enables games and simulations to serve as environments for training intelligent agents.

For this project:

1. Download the "Banana" environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in your GitHub repository (or working directory), in the `p1_navigation/` folder, and unzip (or decompress) the file. 

# Instruction

'p1_navigation' is composed of :

Three notebooks, each one implementing a specific agent to solve the "Banana game":

- Navigation.ipynb : this notebook implements a DQN agent [pdf](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf);
- Navigation-double-dqn.ipynb : this notebook implements a double-DQN agent [arxiv](https://arxiv.org/abs/1509.06461);
- Navigation-dueling-dqn.ipynb : this notebook implements a dueling-DQN agent [arxiv](https://arxiv.org/abs/1511.06581).

A document Report.pdf describing my experiments as well as the algorithms

The weights for each agents
- DQN : checkpoint_dqn_196_2.pth
- DDQN : checkpoint_ddqn_256.pth
- Dueling-DQN : checkpoint_deling_dqn_3.pth

