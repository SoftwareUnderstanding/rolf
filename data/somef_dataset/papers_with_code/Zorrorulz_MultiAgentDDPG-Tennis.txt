# Multi-Agent Deep Deterministic Policy Gradient (MADDPG) -Tennis
### Overview

This project was developed as part of Udacity Deep Reinforcement Learning Nanodegree course. This project solves Tennis environment by training the agent to using Multi-Agent Deep Deterministic Policy Gradient (MADDPG) algorithm. The environment is based on [Unity ML agents](https://github.com/Unity-Technologies/ml-agents). In this environment, two agents bounce a ball with the help of actions taken on the rackets. If an agent hits the ball, it receives a reward of +0.1. If agent does not hit the ball, it receives a reward of -0.01. The environment is considered solved if the agent can keep the ball in play and scores a reward of +0.5. 

### Introduction

For this project, the unity ML [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment is used . The agent is trained to play [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) 



[![Unity ML-Agents Tennis Environment](https://video.udacity-data.com/topher/2018/May/5af7955a_tennis/tennis.png)Unity ML-Agents Tennis Environment](https://classroom.udacity.com/nanodegrees/nd893-ent/parts/0ba70f95-986b-400c-9b2e-59366cca2a49/modules/83e3a45a-a815-4dca-82bc-c6f1b46ac8cd/lessons/c03538e3-4024-41c5-9baa-3be2d91f250c/concepts/da65c741-cdeb-4f34-bb56-d8977385596e#)



In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

### Getting Started

### Installation and Dependencies

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6.

   - **Linux** or **Mac**:

   ```
   conda create --name drlnd python=3.6
   source activate drlnd
   ```

   - **Windows**:

   ```
   conda create --name drlnd python=3.6 
   activate drlnd
   ```

2. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.

   - Next, install the **classic control** environment group by following the instructions [here](https://github.com/openai/gym#classic-control).
   - Then, install the **box2d** environment group by following the instructions [here](https://github.com/openai/gym#box2d).

3. Clone the repository (if you haven't already!), and navigate to the `python/` folder. Then, install several dependencies.

```
git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
pip install .
```

1. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.

```
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

1. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu.

[![Kernel](https://user-images.githubusercontent.com/10624937/42386929-76f671f0-8106-11e8-9376-f17da2ae852e.png)](https://user-images.githubusercontent.com/10624937/42386929-76f671f0-8106-11e8-9376-f17da2ae852e.png)



### Unity Environment Setup:

Unity Environment is already built and made available as part of Deep Reinforcement Learning course at Udacity.

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:

2. Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
3. Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
4. Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
5. Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
6. Place the file in the DRLND GitHub repository, in the `p2_continuous-control/` folder, and unzip (or decompress) the file. 

Then, place the file in the `p3_collab-compet/` folder in the DRLND GitHub repository, and unzip (or decompress) the file.

(*For Windows users*) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

(*For AWS*) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment. You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent. (*To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above.*)



### Multi Agent Deep Deterministic Policy Gradient Algorithm

Please refer to https://arxiv.org/pdf/1706.02275.pdf for understanding of Multi-Agent Deep Deterministic Policy Gradient Algorithm. 

MADDPG is a model free off-policy actor-critic multi-agent algorithm that learns directly from observation spaces. Agent trains itself using the local observation and decides the next best course of action, while critic evaluates the quality of the action by looking at all the observations and actions taken by other agents.

![image-20200407135116052](images/image-20200407135116052.png)

Source: https://arxiv.org/pdf/1706.02275.pdf

The algorithm is listed below:

![image-20200407135738403](images/image-20200407135738403.png)

### Repository

The repository contains the below files:

- final_maddpg_xreplay.ipynb :  Implementation of MADDPG with experienced replay buffer. Training the  agent is implemented here.

- checkpoint_actor1_xreplay.pth : Learned model weights for Agent 1

- checkpoint_actor2_xreplay.pth : Learned model weights for Agent 2 

- checkpoint_critic1_xreplay.pth : Learned model weights for Critic 1

- checkpoint_critic2_xreplay.pth : Learned model weights for Critic 2

- images  directory: contains images used in documentation

- models directory : Contains other working models and work in progress environments used to solve the Tennis environment  

  Before running the ipynb file,Please copy Tennis environment  to this location or modify the filepath in final_maddpg_xreplay.ipynb to point to the correct location.



## Model Architecture:

Pendulum-v0 environment with [Deep Deterministic Policy Gradients (DDPG)](https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/DDPG.ipynb) is used as reference  to build the model.  The model architecture that is used is:

Actor:
	Input(state size of 24) &rarr; Dense Layer(64) &rarr; RELU &rarr; Dense Layer(64) &rarr; RELU &rarr; Dense Layer( action size of 2) &rarr; TANH

Critic:
	Input(state size of 48) &rarr; Dense Layer(64) &rarr; LeakyRELU & actions (4) &rarr; Dense Layer(64) &rarr; Leaky RELU &rarr;  Q value

Agent:
	Actor Local and Critic Local networks are trained and updates the Actor Target and Critic Target networks using weighting factor Tau.

Please refer to Report.md for more details on the model and parameters used for tuning

## Results:

Agent trained with MA-DDPG  with experience replay solved the environment in 1045 episodes.

##### MADDPG with  Experience Replay

![image-20200414084241521](images/image-20200414084241521.png)





