[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Navigation with Deep Q-Network

In this project, I trained an agent to navigate and collect bananas in a large, square world.

## Reinforcement Learning Environment

Unity Machine Learning Agents (ML-Agents) is an open-source Unity plugin that enables games and simulations to serve as environments for training intelligent agents. The gif below shows the environment for this project.

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

## Deep Q-Networks

### 1. Learning algorithm
  - #### Value-based Deep Reinforcement Learning
    - Reinforcement Learning (RL) is a branch of Machine Learning, where an agent outputs an action and the environment returns an observation (the state of the system) and a reward. The goal of an agent is to determine the best action to take and maximizes the overall or total reward.
    - Value-based Deep RL uses nonlinear function approximators (Deep Neural Network) to calculate the value actions based directly on observation from the environment. Deep Learning can be used to find the optimal parameters for these function approximators.
  - #### Experience Replay
    - I created a ReplayBuffer Class to enable experience replay<sup>1, 2</sup>. Using the replay pool, the behavior distribution is averaged over many of its previous states, smoothing out learning and avoiding oscillations. The advantage is that each step of the experience is potentially used in many weight updates.

### 2. Model architecture for the neural network
  - #### Fixed Q-Targets
    - I adopted Double Deep Q-Network structure<sup>1, 2</sup> with three fully connected layers. If a single network is used, the Q-functions values change at each step of training, and then the value estimates can quickly spiral out of control. I used a target network to represent the old Q-function, which is used to compute the loss of every action during training.

### 3. Hyperparameters

  - #### Replay buffer size
    - BUFFER_SIZE = int(1e5)
  - #### Minibatch size
    - BATCH_SIZE = 64
  - #### Discount factor
    - GAMMA = 0.99
  - #### For soft update of target parameters
    - TAU = 1e-3
  - #### Learning rate
    - LR = 5e-4
  - #### How often to update the network
    - UPDATE_EVERY = 4
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
    git clone https://github.com/iDataist/Navigation-with-Deep-Q-Network
    ```
    c. Create local environment

    - Create (and activate) a new environment, named `dqn-env` with Python 3.7. If prompted to proceed with the install `(Proceed [y]/n)` type y.

        - __Linux__ or __Mac__:
        ```
        conda create -n dqn-env python=3.7
        conda activate dqn-env
        ```
        - __Windows__:
        ```
        conda create --name dqn-env python=3.7
        conda activate dqn-env
        ```

        At this point your command line should look something like: `(dqn-env) <User>:USER_DIR <user>$`. The `(dqn-env)` indicates that your environment has been activated, and you can proceed with further package installations.

    - Install a few required pip packages, which are specified in the requirements text file. Be sure to run the command from the project root directory since the requirements.txt file is there.
        ```
        pip install -r requirements.txt
        ipython3 kernel install --name dqn-env --user
        ```
    - Open Jupyter Notebook, and open the Navigation.ipynb file.
        ```
        jupyter notebook
        ```
2. Download the Unity Environment

   Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

## File Descriptions

1. [requirements.txt](https://github.com/iDataist/Navigation-with-Deep-Q-Network/blob/main/requirements.txt) - Includes all the required libraries for the Conda Environment.
2. [model.py](https://github.com/iDataist/Navigation-with-Deep-Q-Network/blob/main/model.py) - Defines the QNetwork which is the nonlinear function approximator to calculate the value actions based directly on observation from the environment.
3. [dqn_agent.py](https://github.com/iDataist/Navigation-with-Deep-Q-Network/blob/main/dqn_agent.py) -  Defines the Agent that uses Deep Learning to find the optimal parameters for the function approximators, determines the best action to take and maximizes the overall or total reward.
4. [Navigation.ipynb](https://github.com/iDataist/Navigation-with-Deep-Q-Network/blob/main/Navigation.ipynb) - The main file that trains the Deep Q-Network and shows the trained agent in action. This file can be run in the Conda environment.

## Plot of Rewards
The environment was solved in 463 episodes, with the average reward score of 13 to indicate solving the environment.

![](score.png)

## Ideas for Future Work

- Prioritized Experience Replay<sup>3</sup>: I have adopted experience replay in the DQN. But some of these experiences may be more important for learning than others. Moreover, these important experiences might occur infrequently. If we sample the batches uniformly, then these experiences have a very small chance of getting selected. Since buffers are practically limited in capacity, older important experiences may get lost. I will implement prioritized experience replay<sup>4</sup> will help to optimize the selection of experiences.

- Dueling Networks<sup>4</sup>: Dueling networks use two streams, one that estimates the state value function and one that estimates the advantage for each action.These streams may share some layers in the beginning, then branch off with their own fully-connected layers. The desired Q values are obtained by combining the state and advantage values. The value of most states don't vary a lot across actions. So, it makes sense to try and directly estimate them. But we still need to capture the difference actions make in each state. This is where the advantage function comes in.

References:
1. Riedmiller, Martin. "Neural fitted Q iteration–first experiences with a data efficient neural reinforcement learning method." European Conference on Machine Learning. Springer, Berlin, Heidelberg, 2005. http://ml.informatik.uni-freiburg.de/former/_media/publications/rieecml05.pdf
2. Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning." Nature518.7540 (2015): 529. http://www.davidqiu.com:8888/research/nature14236.pdf
3. Schaul, Quan, et al. "Prioritized Experience Replay." ICLR (2016). https://arxiv.org/abs/1511.05952
4. Wang, Schaul, et al. "Dueling Network Architectures for Deep Reinforcement Learning." 2015. https://arxiv.org/abs/1511.06581


