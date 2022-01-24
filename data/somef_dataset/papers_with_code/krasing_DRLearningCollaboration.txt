[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135602-b0335606-7d12-11e8-8689-dd1cf9fa11a9.gif "Trained Agents"

# Train an Agent to Play Tennis

This repository contains material related to the **Collaboration and Competition** project of the Udacity's [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) program.  

![Trained Agents](./images/tennis.png)



## Project Details

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. A sequence of 3 observations are stacked together. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single score for each episode.

The perfect score is about +2.7 (task dutation of 1000 steps but single pass takes multiple steps).

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

    Download [this file](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip) for Linux operating system. Decompress in the same folder where the project files are. This project uses rich simulation environments from [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents) but you will not need to install Unity - the environment is already built into the downloaded file.

## Run the notebook

1. Execute in terminal `jupyter notebook Tennis-old.ipynb` to load the notebook. Make sure that *ddpg_agent_old.py* and *model.py* are in the same folder.

2. Before running code in the notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu. 

![Kernel](./images/kernel.png)

3. To train the Agent execute:
  - Sections 1. Start the Environment
  - Section 2. Examine the State and Action Spaces
  - Section 3. Train the Agent with DDPG 
  
4. To evaluate the Agent performance and fine tune:
  - Sections 1. Start the Environment
  - Section 4. Evaluate score over 100 games
  - Section 5. More training for the initial phase of the game 
  - Section 6. Evaluate score over 100 games again
  
4. To watch the Game execute:
  - Sections 1. Start the Environment
  - Section 7. Demo play

## Read some explanations

[Report](./Report.md)


## References

* The Udacity's [Deep Reinforcement Learning Nanodegree program](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) repository, https://github.com/udacity/deep-reinforcement-learning

* Lillicrap T.P., Hunt J.J., Pritzel A., et.al., Continuous control with deep reinforcement learning, arXiv:1509.02971v5, https://arxiv.org/abs/1509.02971

* Implementations of the **Deep Deterministic Policy Gradients** in OpenAI Gym environments:
    - [Pendulum](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum): Use OpenAI Gym's Pendulum environment.
    - [BipedalWalker](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-bipedal): Use OpenAI Gym's BipedalWalker environment.

* The Unity ML-Agents toolkit example learning environments, https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis

* Lowe R., Yi Wu, Tamar A., Harb J., Abbeel P., Mordatch I., Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments, 2017, https://arxiv.org/abs/1706.02275
