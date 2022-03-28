# Tennis With Multi Agent Reinforcement

In this project, I trained two agents to play tennis.

## Reinforcement Learning Environment

Unity Machine Learning Agents (ML-Agents) is an open-source Unity plugin that enables games and simulations to serve as environments for training intelligent agents. The image below shows the [tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment for this project.

![](tennis.png)

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents).

## Deep Deterministic Policy Gradient

### 1. Learning Algorithm

  - DDPG<sup>1</sup> is a different kind of actor-critic method. It could be seen as an approximate DQN instead of an actual actor-critic. This is because the critic in DDPG is used to approximate the maximizer over the Q values of the next state and not as a learned baseline.
  - One of the DQN agent's limitations is that it is not straightforward to use in continuous action spaces. Imagine a DQN network that takes the state and outputs the action-value function. For example, for two actions, say, up and down, ```Q(S, "up")``` gives you the estimated expected value for selecting the up action in state ```S```, say ```-2.18```. ```Q(S,  "down")``` gives you the estimated expected value for choosing the down action in state ```S```, say ```8.45```. To find the max action-value function for this state, you just calculate the maximum of these values. Pretty easy. It's straightforward to do a ```max``` operation in this example because this is a discrete action space. Even if you had more actions say a left, a right, a jump, and so on, you still have a discrete action space. Even if it were high dimensional with many, many more actions, it would still be feasible. But how do you get the value of continuous action with this architecture? Say you want the jump action to be continuous, a variable between ```1``` and ```100``` centimeters. How do you find the value of jump, say ```50``` centimeters? This is one of the problems DDPG solves.
  - In DDPG, we use two deep neural networks: the actor and the critic.
    - The actor here is used to approximate the optimal policy deterministically. That means we want to always output the best-believed action for any given state. This is unlike stochastic policies in which we want the policy to learn a probability distribution over the actions. In DDPG, we want the believed the best action every single time we query the actor network. That is a deterministic policy. The actor is learning the ```argmax Q(S, a)```, which is the best action.
    - The critic learns to evaluate the optimal action-value function by using the actor's best-believed action. Again, we use this actor, an approximate maximizer, to calculate a new target value for training the action-value function, much like DQN does.
  - How to adapt the single-agent auto techniques to the multi-agent case?
    - The simplest approach should be to train all the agents independently without considering the existence of other agents. In this approach, any agent considers all the others to be a part of the environment and learns its own policy. Since all are learning simultaneously, the environment as seen from the prospective of a single agent, changes dynamically. This condition is called non-stationarity of the environment. In most single agent algorithms, it is assumed that the environment is stationary, which leads to certain convergence guarantees. Hence, under non-stationarity conditions, these guarantees no longer hold.
    - The second approach is the multi agent approach. The multi agent approach takes into account the existence of multiple agents. Here, a single policy is lowered for all the agents. It takes as input the present state of the environment and returns the action of each agent in the form of a single joint action vector. The joint action space would increase exponentially with the number of agents. If the environment is partially observable or the agents can only see locally, each agent will have a different observation of the environment state, hence it will be difficult to disambiguate the state of the environment from different local observations. So this approach works well only when each agent knows everything about the environment.

### 2. Model Architecture for the Neural Network
  - #### Actor Network
    |Layer        | Input/Output Sizes | Activation Function      |
    | ----------- | -----------        | -----------              |
    | Linear      | (24, 128)          | Leaky-relu               |
    | Linear      | (128, 128)         | Leaky-relu               |
    | Linear      | (128, 128)         | Leaky-relu               |
    | Linear      | (128, 2)           | Tanh                     |

  - #### Critic Network
    |Layer        | Input/Output Sizes | Activation Function      |
    | ----------- | -----------        | -----------              |
    | Linear      | (48, 128)          | Leaky-relu               |
    | Linear      | (132, 128)         | Leaky-relu               |
    | Linear      | (132, 128)         | Leaky-relu               |
    | Linear      | (128, 1)           |                          |

### 3. Hyperparameters
  - #### Replay buffer size
    - BUFFER_SIZE = int(1e5)
  - #### Minibatch size
    - BATCH_SIZE = 128
  - #### Discount factor
    - GAMMA = 0.99
  - #### For soft update of target parameters
    - TAU = 1e-3
  - #### Learning rate
    - LR = 5e-4
  - #### learning rate of the actor
    - LR_ACTOR = 1e-5
  - #### learning rate of the critic
    - LR_CRITIC = 1e-4
  - #### L2 weight decay
    - WEIGHT_DECAY = 0
  - #### update every UPDATE_EVERY time steps
    - UPDATE_EVERY = 1

## Getting Started

1. Create the Conda Environment

    a. Install [`miniconda`](http://conda.pydata.org/miniconda.html) on your computer, by selecting the latest Python version for your operating system. If you already have `conda` or `miniconda` installed, you should be able to skip this step and move on to step b.

    **Download** the latest version of `miniconda` that matches your system.

    |        | Linux | Mac | Windows |
    |--------|-------|-----|---------|
    | 64-bit | [64-bit (bash installer)][lin64] | [64-bit (bash installer)][mac64] | [64-bit (exe installer)][win64]
    | 32-bit | [32-bit (bash installer)][lin32] |  | [32-bit (exe installer)][win32]

    [win64]: https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86_64.exe
    [win32]: https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86.exe
    [mac64]: https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
    [lin64]: https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
    [lin32]: https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86.sh

    **Install** [miniconda](http://conda.pydata.org/miniconda.html) on your machine. Detailed instructions:

    - **Linux:** http://conda.pydata.org/docs/install/quick.html#linux-miniconda-install
    - **Mac:** http://conda.pydata.org/docs/install/quick.html#os-x-miniconda-install
    - **Windows:** http://conda.pydata.org/docs/install/quick.html#windows-miniconda-install

    b. Install git and clone the repository.

    For working with Github from a terminal window, you can download git with the command:
    ```
    conda install git
    ```
    To clone the repository, run the following command:
    ```
    cd PATH_OF_DIRECTORY
    git clone hhttps://github.com/iDataist/Tennis-With-Multi-Agent-Reinforcement
    ```
    c. Create local environment

    - Create (and activate) a new environment, named `maddpg-env` with Python 3.7. If prompted to proceed with the install `(Proceed [y]/n)` type y.

        - __Linux__ or __Mac__:
        ```
        conda create -n maddpg-env python=3.7
        conda activate maddpg-env
        ```
        - __Windows__:
        ```
        conda create --name maddpg-env python=3.7
        conda activate maddpg-env
        ```

        At this point your command line should look something like: `(maddpg-env) <User>:USER_DIR <user>$`. The `(maddpg-env)` indicates that your environment has been activated, and you can proceed with further package installations.

    - Install a few required pip packages, which are specified in the requirements text file. Be sure to run the command from the project root directory since the requirements.txt file is there.
        ```
        pip install -r requirements.txt
        ipython3 kernel install --name maddpg-env --user
        ```
    - Open Jupyter Notebook, and open the Continuous_Control.ipynb file. Run all the cells in the jupyter notebook to train the agents.
        ```
        jupyter notebook
        ```
2. Download the Unity Environment

   a. Download the environment from one of the links below.  You need only select the environment that matches your operating system:

    - Linux: click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

    (For Windows users) Check out this [link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (For AWS) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use this [link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment. You will not be able to watch the agent without enabling a virtual screen, but you will be able to train the agent. (To watch the agent, you should follow the instructions to enable a virtual screen, and then download the environment for the Linux operating system above.)

    b. Place the file in the folder with the jupyter notebook, and unzip (or decompress) the file.

## File Descriptions

1. [requirements.txt](https://github.com/iDataist/Tennis-With-Multi-Agent-Reinforcement/blob/main/requirements.txt) - Includes all the required libraries for the Conda Environment.
2. [model.py](https://github.com/iDataist/Tennis-With-Multi-Agent-Reinforcement/blob/main/model.py) - Defines the actor and critic networks.
3. [agent.py](https://github.com/iDataist/Tennis-With-Multi-Agent-Reinforcement/blob/main/agent.py) - Defines the Agent that uses MADDPG to determine the best action to take and maximizes the overall or total reward.
4. [Tennis.ipynb](https://github.com/iDataist/Tennis-With-Multi-Agent-Reinforcement/blob/main/Tennis.ipynb) - The main file that trains the agents. This file can be run in the Conda environment.

## Plot of Rewards
The environment was solved in 728 episodes.

![](score.png)

## Ideas for Future Work

- Distributed Prioritized Experience Replay<sup>5</sup>
- Reinforcement Learning with Prediction-Based Rewards<sup>6</sup>
- Proximal Policy Optimization<sup>7</sup>
- OpenAI Five<sup>8</sup>
- Curiosity-driven Exploration by Self-supervised Prediction<sup>9</sup>

References:

1. Lillicrap, Hunt, et al. "Continuous control with deep reinforcement learning." 2015. https://arxiv.org/abs/1509.02971

2. Riedmiller, Martin. "Neural fitted Q iteration–first experiences with a data efficient neural reinforcement learning method." European Conference on Machine Learning. Springer, Berlin, Heidelberg, 2005. http://ml.informatik.uni-freiburg.de/former/_media/publications/rieecml05.pdf

3. Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning." Nature518.7540 (2015): 529. http://www.davidqiu.com:8888/research/nature14236.pdf

4. Mnih,  Kavukcuoglu, et al. "Playing Atari with Deep Reinforcement Learning." https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

5. Schaul, Quan, et al. "Prioritized Experience Replay." ICLR (2016). https://arxiv.org/abs/1511.05952

6. https://openai.com/blog/reinforcement-learning-with-prediction-based-rewards/

7. https://openai.com/blog/openai-baselines-ppo/

8. https://openai.com/blog/openai-five/

9.  https://pathak22.github.io/noreward-rl/