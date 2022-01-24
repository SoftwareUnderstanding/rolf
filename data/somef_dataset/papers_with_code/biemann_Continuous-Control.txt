[//]: # (Image References)

[image1]: https://github.com/biemann/Continuous-Control/blob/master/bin/solved_in_80.png "Solved"
[image2]: https://github.com/biemann/Continuous-Control/blob/master/bin/Reacher.gif "Gif"


# Continuous Control

The first two sections are similar to the ones described in the project description. In our case, the agent has been trained using Mac OSX, so you may have to change the path dependencies in our code, following the instructions in the repository https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control. We recommend following the instructions there and run the notebook to test whether the environments (Unity and Anaconda) have been properly installed. You will need to activate this environment in order to make the code work.

We decided to train the first version of the environment described there with only one agent.

The report describing our experiments and the architecture lies further down in the Readme file.

## Introduction

For this project, we will work with the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

The task is episodic, and in order to solve the environment,  the agent must get an average score of +30 over 100 consecutive episodes.

## Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

2. Place the file in the DRLND GitHub repository, in the `p2_continuous-control/` folder, and unzip (or decompress) the file. 

## Training an Agent from scratch

If you want to train an agent, move inside the downloaded repository and type 

```
python train.py
```
in the terminal (after having activated the correct python environment).

We can expect to solve the episode in 80-250 episodes using the original hyperparameters and network architecture. If you want 
to test other parameters, you would have to change the parameters at the top of the respective classes.

## Testing the Agent

In order to see a trained agent, you can type in the terminal

```
python test.py
```
We added two checkpoints, both located in the bin folder: `checkpoint_finished.pth`, that is the result of the actor network we get after having solved the task and `checkpoint_critic.pth` for the critic network.

## The DDPG algorithm

To solve the task, we use an implementation the Deep Deterministic Policy Gradient (DDPG) algorithm of the following paper: https://arxiv.org/pdf/1509.02971.pdf. The code is heavily inspired by the example used to solve the OpenAI-gym Pendulum task. Most of our work here was to adapt this algorithm to our Unity environment and especially modifying the hyperparameters to make the agent learn something.

The idea of the DDPG algorithm is an improvement of the Q-learning algorithm (see our previous Navigation project). The basic ideas are the same, as it also uses a replay buffer to reuse past experiments (to avoid a heavy correleation between subsequent states), as well as two parallel networks : the target and local network that do not update simultaneously in order to combat divergence of the algorithm. However, Q-learning is a value-based method and is not well-suited for continuous tasks (such as this one) and discrete tasks with high-dimensional action spaces. We also use soft updates, where the weights of the local network are slowly mixed into the target network.

Thie idea is to use policy-based method instead, using a critic network. The actor network approximates the optimal policy deterministically. The critic network is evaluating the actions taken by the actor network and tries to predict the reward of the chosen actions. The actor policy is updated using the sampled policy gradient.

In addition, in order to encourage exploration, they also implemented the Ornstein-Uhlenbeck noise. For more details, we refer to the original paper or to understand how it is implemented to the code.

## Implementation of the DDPG algorithm

The `ddpg_agent.py`class is an implementation of the DDPG algorithm following the DeepMind paper as exposed in this nanodegree. The only slight modification we did was an implementation of a learning rate scheduler in order to help solving the task faster. 

The `model.py`is the neural network the DDPG algorithm uses to make the agent and the critic learn. The architectures will be described in the following section.

The `train.py`is the main function of our project. It adapts the DDPG algorithm to this particular environment. The code follows the notebook https://github.com/udacity/deep-reinforcement-learning/blob/master/p2_continuous-control/Continuous_Control.ipynb. Note that we slightly simplified the code, because this example used a version with 20 agents, whereas we only trained one agent.

## Network architecture and Hyperparameters

In contrast to the paper or the example, we used two different architectures for the actor and critc network. The actor network consists of 3 layers of 128 neurons each, followed by the tanh activation function (in order to have values between -1 and 1). The critc network is built with 3 layers of 64 neurons each. We use the selu activation function, instead of the more standard relu for both networks. In addition, we use batch normalisation only for the actor network (using it for both significantly hurt the training). 

We expermiented with the tau parameter (for soft updates), the gamma parameter (reward discount) and sigma (for data exploration). We finally settled for tau = 5e-4, gamma = 0.9 and sigma = 0.12, all three being lower than the original parameters. For the learning rate, we started for both networks with 5e-4 and divided it by 2 every 100 episodes, using a learning rate scheduler. Note that one episode consists of 1000 timesteps in our case, which is larger than for the Pendelum example. When lowering this parameter, the task will be solved in less episodes, but the same amount of time.

## Results

Initally, we had quite some problems to make the agent learn, concluding it is better to begin with a simple architecture and only implementing methods, such as Batch Normalisation once it was learning something. 

We were able to solve the task in 80 episodes (it took 180 episodes to get an average of 30.0 over the last 100 episodes). However, the algorithm is unstable and with the same parameters, we may need far more episodes. However, we expect that the task will be solved in less than 300 episodes.

We have here a graph, showing the learning of our best result. 

![Solved][image1]

We note that it reached the desired score of 30 consistently after 130 episodes and was being closed to 40 at the end of the training. The trend is quite similar to what we saw with the Q-learning algorithm : it has quite some trouble to learn anything at the beginning, learns fast in the middle and reaches a plateau at the end, because it learned the task (the maximum score should be around 40).

![Gif][image2]

We added a gif showing a trained agent. This agent reached a score of 38.0 and follows the ball pretty well during the whole episode.

## Future Work

In future, we could compare this method to other policy based methods, such as A2C, taking advantage of the parallelisation of different agents. We can also try this algorithm on more challenging environments, such as the crawler environment. This will require to optimise the architecture even more carefully to make the agent learn something. We would like to investigate such examples in future.

