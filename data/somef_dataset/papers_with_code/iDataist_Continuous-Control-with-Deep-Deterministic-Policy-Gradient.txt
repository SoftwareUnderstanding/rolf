# Continuous Control with Deep Deterministic Policy Gradient

In this project, I trained twenty double-jointed arms to move to target locations.

## Reinforcement Learning Environment

Unity Machine Learning Agents (ML-Agents) is an open-source Unity plugin that enables games and simulations to serve as environments for training intelligent agents. The gif below shows the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment for this project.

![](reacher.gif)

In this environment, twenty double-jointed arms can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of the agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1. The agents must get an average score of +30 (over 100 consecutive episodes, and over all agents) to solve the environment (version 1 or 2).

## Deep Deterministic Policy Gradient

### 1. Learning Algorithm

  - DDPG<sup>1</sup> is a different kind of actor-critic method. It could be seen as an approximate DQN instead of an actual actor-critic. This is because the critic in DDPG is used to approximate the maximizer over the Q values of the next state and not as a learned baseline.
  - One of the DQN agent's limitations is that it is not straightforward to use in continuous action spaces. Imagine a DQN network that takes the state and outputs the action-value function. For example, for two actions, say, up and down, ```Q(S, "up")``` gives you the estimated expected value for selecting the up action in state ```S```, say ```-2.18```. ```Q(S,  "down")``` gives you the estimated expected value for choosing the down action in state ```S```, say ```8.45```. To find the max action-value function for this state, you just calculate the maximum of these values. Pretty easy. It's straightforward to do a ```max``` operation in this example because this is a discrete action space. Even if you had more actions say a left, a right, a jump, and so on, you still have a discrete action space. Even if it were high dimensional with many, many more actions, it would still be feasible. But how do you get the value of continuous action with this architecture? Say you want the jump action to be continuous, a variable between ```1``` and ```100``` centimeters. How do you find the value of jump, say ```50``` centimeters? This is one of the problems DDPG solves.
  - In DDPG, we use two deep neural networks: the actor and the critic.
    - The actor here is used to approximate the optimal policy deterministically. That means we want to always output the best-believed action for any given state. This is unlike stochastic policies in which we want the policy to learn a probability distribution over the actions. In DDPG, we want the best action every single time we query the actor network. That is a deterministic policy. The actor is learning the ```argmax Q(S, a)```, which is the best action.
    - The critic learns to evaluate the optimal action-value function by using the actor's best-believed action. Again, we use this actor, an approximate maximizer, to calculate a new target value for training the action-value function, much like DQN does.

### 2. Model Architecture for the Neural Network
  - #### Actor Network
    |Layer        | Input/Output Sizes | Activation Function |
    | ----------- | -----------        | -----------         |
    | Linear      | (33, 128)          | Relu                |
    | Linear      | (128, 128)         | Relu                |
    | Linear      | (128, 4)           | Tanh                |

  - #### Critic Network
    |Layer        | Input/Output Sizes | Activation Function |
    | ----------- | -----------        | -----------         |
    | Linear      | (33, 128)          | Relu                |
    | Linear      | (132, 128)         | Relu                |
    | Linear      | (128, 1)           |                     |

  - #### Experience Replay
    - I created a ReplayBuffer Class to enable experience replay<sup>2, 3</sup>. Using the replay pool, the behavior distribution is averaged over many of its previous states, smoothing out learning and avoiding oscillations. The advantage is that each step of the experience is potentially used in many weight updates.
  - #### Soft Updates to The Target Networks
    - In DQN, there are two copies of the network weights, the regular and the target network. In the Atari paper<sup>4</sup> in which DQN was introduced, the target network is updated every 10,000 time steps. We can simply copy the weights of the regular network into the target network. The target network is fixed for 10,000 time steps, and then gets a big update.
    - In DDPG, there are two copies of the network weights for each network: a regular for the actor, a regular for the critic, a target for the actor, and a target for the critic. The target networks are updated using a soft updates strategy. A soft update strategy consists of slowly blending your regular network weights with the target network weights. So, every time step, I make the target network be 99.99 percent of the target network weights and only 0.01 percent of the regular network weights. I slowly mix in the regular network weights into the target network weights. The regular network is the most up-to-date network, while the target network is the one we use for prediction to stabilize training. We get faster convergence by using this update strategy. Soft updates can be used with other algorithms that use target networks, including DQN.

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
    - LR_ACTOR = 1e-3
  - #### learning rate of the critic
    - LR_CRITIC = 1e-3
  - #### L2 weight decay
    - WEIGHT_DECAY = 0

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
    git clone https://github.com/iDataist/Continuous-Control-with-Deep-Deterministic-Policy-Gradient
    ```
    c. Create local environment

    - Create (and activate) a new environment, named `ddpg-env` with Python 3.7. If prompted to proceed with the install `(Proceed [y]/n)` type y.

        - __Linux__ or __Mac__:
        ```
        conda create -n ddpg-env python=3.7
        conda activate ddpg-env
        ```
        - __Windows__:
        ```
        conda create --name ddpg-env python=3.7
        conda activate ddpg-env
        ```

        At this point your command line should look something like: `(ddpg-env) <User>:USER_DIR <user>$`. The `(ddpg-env)` indicates that your environment has been activated, and you can proceed with further package installations.

    - Install a few required pip packages, which are specified in the requirements text file. Be sure to run the command from the project root directory since the requirements.txt file is there.
        ```
        pip install -r requirements.txt
        ipython3 kernel install --name ddpg-env --user
        ```
    - Open Jupyter Notebook, and open the Continuous_Control.ipynb file. Run all the cells in the jupyter notebook to train the agents.
        ```
        jupyter notebook
        ```
2. Download the Unity Environment

   a. Download the environment from one of the links below.  You need only select the environment that matches your operating system:

   - Version 1: One (1) Agent
       - Linux: click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
       - Mac OSX: click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
       - Windows (32-bit): [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
       - Windows (64-bit): [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)
   - Version 2: Twenty (20) Agents
       - Linux: click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
       - Mac OSX: click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
       - Windows (32-bit): click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
       - Windows (64-bit): click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

    (For Windows users) Check out this [link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (For AWS) If you'd like to train the agent on AWS (and have not enabled a virtual screen), then please use this [link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or this [link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment. You will not be able to watch the agent without enabling a virtual screen, but you will be able to train the agent. (To watch the agent, you should follow the instructions to enable a virtual screen, and then download the environment for the Linux operating system above.)

    b. Place the file in the folder with the jupyter notebook, and unzip (or decompress) the file.

## File Descriptions

1. [requirements.txt](https://github.com/iDataist/Continuous-Control-with-Deep-Deterministic-Policy-Gradient/blob/main/requirements.txt) - Includes all the required libraries for the Conda Environment.
2. [model.py](https://github.com/iDataist/Continuous-Control-with-Deep-Deterministic-Policy-Gradient/blob/main/model.py) - Defines the actor and critic networks.
3. [ddpg_agent.py](https://github.com/iDataist/Continuous-Control-with-Deep-Deterministic-Policy-Gradient/blob/main/ddpg_agent.py) -  Defines the Agent that uses DDPG to determine the best action to take and maximizes the overall or total reward.
4. [Continuous_Control.ipynb](https://github.com/iDataist/Continuous-Control-with-Deep-Deterministic-Policy-Gradient/blob/main/Continuous_Control.ipynb) - The main file that trains the actor and critic networks. This file can be run in the Conda environment.

## Plot of Rewards
The environment was solved in 114 episodes.

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
