[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135602-b0335606-7d12-11e8-8689-dd1cf9fa11a9.gif "Trained Agents"
[image2]: images/kernel.png "Kernel"

# Train an robotic arm to reach target locations

This repository contains material related to the **Continuous Control** project of the Udacity's [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) program.  

![Trained Agents](./images/scene.png)



## Project Details

In this environment, a double-jointed arm can move to target locations. The simulation contains 20 identical agents, each with its own copy of the environment.

A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of the agent is to maintain its position at the target location for as many time steps as possible. The task is episodic, and in order to solve the environment, **the agents must get an average score of +30** (over 100 consecutive episodes, and over all agents). The perfect score is +100 (task dutation of 1000 steps x 0.01 reward each).

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

To solve the problem we have used the resources listed in the References section of this document.

## Dependencies

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6. (__Linux__ or __Mac__)
	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```

2. Clone the repository (if you haven't already!), and navigate to the `python/` folder.  Then, install several dependencies.
```bash
git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
pip install .
```

3. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.  
```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

4. Download the Unity Environment:

    Download [this file](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip) for Linux operating system. Decompress in the same folder where the project files are. This project uses rich simulation environments from [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents) but you will not need to install Unity - the environment is already built into the downloaded file.

5. If you plan to play with the OpenAI gym examples, follow the instructions in [this repository](https://github.com/openai/gym):
    - perform a minimal install of OpenAI gym:
```bash
git clone https://github.com/openai/gym.git
cd gym
pip install -e .
```    
	- install the **classic control** environment group:
```bash
pip install -e '.[classic_control]'
```
	- install the **box2d** environment group:
```bash
pip install -e '.[box2d]'
```

## Run the notebook

1. Execute in terminal `jupyter notebook Continuous_Control.ipynb` to load the notebook. Make sure that *ddpg_agent.py* and *model.py* are in the same folder.

2. Before running code in the notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu. 

  ![Kernel][image2]

3. To train the Agent execute:
  - Sections 1. Start the Environment
  - Section 2. Examine the State and Action Spaces
  - Section 3. Train the Agent with DDPG 
  
4. To watch a Smart Agent execute:
  - Sections 1. Start the Environment
  - Section 4. Watch a Smart Agent!


## References

* The Udacity's [Deep Reinforcement Learning Nanodegree program](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) repository, https://github.com/udacity/deep-reinforcement-learning

* Lillicrap T.P., Hunt J.J., Pritzel A., et.al., Continuous control with deep reinforcement learning, arXiv:1509.02971v5, https://arxiv.org/abs/1509.02971

* Implementations of the **Deep Deterministic Policy Gradients** in OpenAI Gym environments:
    - [Pendulum](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum): Use OpenAI Gym's Pendulum environment.
    - [BipedalWalker](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-bipedal): Use OpenAI Gym's BipedalWalker environment.

* [Cheatsheet](https://github.com/udacity/deep-reinforcement-learning/blob/master/cheatsheet) for reinforcement learning.