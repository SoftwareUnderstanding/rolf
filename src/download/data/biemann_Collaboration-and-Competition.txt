[//]: # (Image References)

[image1]: https://github.com/biemann/Collaboration-and-Competition/blob/master/bin/1039.png "Results"
[image2]: https://github.com/biemann/Collaboration-and-Competition/blob/master/bin/tennis.gif "Trained Agent"


# Project 3: Collaboration and Competition

### Introduction

For this project, we will work with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

2. Place the file in the DRLND GitHub repository, in the `p3_collab-compet/` folder, and unzip (or decompress) the file. 


## Training an Agent from scratch

If you want to train an agent, move inside the downloaded repository and type 

```
python train.py
```
in the terminal (after having activated the correct python environment).

We can expect to solve the episode in 1000-2000 episodes using the original hyperparameters and network architecture. If you want to test other parameters, you would have to change the parameters at the top of the respective classes.

## Testing the Agent

In order to see a trained agent, you can type in the terminal

```
python test.py
```
We added four checkpoints, all located in the bin folder: one actor per agent:`actor0_finished.pth` and `actor1_finished.pth`, that is the result of the actor network we get after having solved the task and `critic0_finished.pth` and `critic1_finished.pth` for the critic network.

## The MADDPG algorithm and its implementation

This algorithm is an extension of the classical DDPG algorithm, as used in following paper: https://arxiv.org/pdf/1706.02275.pdf It extends this algorithm to the multi-agent setting, which is the case here as the two agents need to collaborate to achieve the best possible reward. For more details, we refer to the paper.

Our implementation follows the exercice we had during this nanodegree on multi-agent systems. In contrast to the project, adapting the code to the current environment was more difficult than expected and we refactored a relevant amount of the code to make it work in this setting. 

We devided the code into 4 classes:

The `train.py` class, that is the main class in our project that interacts with the Unity environment. The design of this class is quite similar to what we did before in this nanodegree.

The `maddpg.py`class turns mostly around the update() function that was heavily refactored in comparision to the course example to make it work in this environment. The idea of the maddpg algorithm is that we have a separate mddpg network for each agent.

The `ddpg.py`file contains in fact three classes, that are essential in the algorithm: The `DDPGAgent`class, the `OUNoise`class and the `ReplayBuffer`class. The Ornstein-Uhlenbek noise helps for data exploration and the replay buffer combats a strong colleration between the data. 

The `model.py`class contains our neural network architecture, that is being called by the `DDPGAgent`class.

## Network architecture and hyperparameters

The network architecture is the same than the one of the previous project: We used for the actor networks 2 layers of 128 neurons each, follwed by the tanh activation function. For the critics, we used 2 layers of 64 neurons. We used the selu activation function. In contrast to the continuous control project, we did not use batch normalisation as it hurt the performence massively. We also tried with 128-128 neurons for the critic network and this architecture also solved the task, but we decided to come back to the smaller architecture as the training was more reliable.

We used for both networks a learning of 5e-4. We tried to implement a learning late scheduler without success. Due to the high randomness in learning, we could not predict how the network will behave and so we did not know how to adapt the learning rate. We used a very high tau parameter for the soft updates: 0.02. For the discount factor gamma, we used the relatively high value of 0.99, because we thought that in this environment, it is quite important to maximise the reward in the long run and not shortly (in contrast to say the reacher environment).

The network was very sensitive to the Ornstein-Uhlenbeck noise for data exploration. We tried several approaches to make the algorithm learn reliably or quickly. We finally used a relatively small sigma parameter with 0.05. However, we initialise the value of the noise at 2 in our main function. The idea was to slightly reduce noise every step, so that at the end the noise should be relatively small. This gave us the possibility to accelerate the training process in some cases. However, in some cases the agents failed to learn anything at all with the same hyperparameters. It really depends on how well the agent learns at the beginning. If he fails to learn then, the data exploration is becoming smaller with the time, so the agent will be stuck in this state. That is why in the final version, we did not include noise reduction, so that the agent is able to solve the task reliably. However, note that our best results have been achieved with noise reduction.

## Results

We show here the graph of the agents that were able to solve the task in 1039 episodes (it took 1139 episodes to reach an average score of 0.5 over the last 100 episodes). As described above, the task has been solved using noise reduction.

![solved][image1]

The graph is relatively interesting, as the agents are not able to learn anything at the beginning (sometimes the ball passes once over the net but not more). Past 900 episodes, the agents begin to learn very quickly. Note that the better the agents become, the longer the episodes will be, so that the agents will be able to extract more information on longer episodes than short ones (in contrast, the two previous had a fix length). This may explain the exponential behaviour of this graph.

Note that this graph is quite different to the graphs of the two previous projects, where the agent learns slowly at the beginning, fast in the middle and slowly again at the end. Also the length of the episodes are far shorter than in the other tasks, so that explains why the number of required episodes is higher. We wanted to solve the task in less than 1000 episodes, but failed to do so. Using the actual parameters, you should expect to solve the task in between 1000 and 2000 episodes.

As an illustration of how our trained agents, that achieved a score of 1.8, behave, we show the following gif:

![Trained Agent][image2]

## Future work

In future, we would like to train similar tasks in even more challenging environments, such as the soccer environment. We also would like to train the agents in some competitive game, using the Alpha Zero algorithm. This algorithm (and Alpha Go) got us interested in Reinforcement Learning and we hope being able to implement this in a personal project.
