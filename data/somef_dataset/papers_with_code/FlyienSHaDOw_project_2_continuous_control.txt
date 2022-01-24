# Continuous Control

---
## 1. Project Details

In this environment, a double-jointed arm can move to target locations. A reward of `+0.1` is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of `33` variables corresponding to position, rotation, velocity, and angular velocities of the arm.  Each action is a vector with four numbers, corresponding to torque applicable to two joints.  Every entry in the action vector must be a number between `-1` and `1`.

It is required that we can achieve an average score of +30.

## 2. Start the Environment

We begin by importing the necessary packages.  If the code cell below returns an error, please revisit the project instructions to double-check that you have installed [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md) and [NumPy](http://www.numpy.org/).


```python
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from unityagents import UnityEnvironment
import numpy as np
import random
import copy
from collections import namedtuple, deque
import os
import time
import sys

from time import sleep
import matplotlib.pyplot as plt

device = torch.device("cpu")
```

Next, we will start the environment!  **_Before running the code cell below_**, change the `file_name` parameter to match the location of the Unity environment that you downloaded.

- **Mac**: `"path/to/Reacher.app"`
- **Windows** (x86): `"path/to/Reacher_Windows_x86/Reacher.exe"`
- **Windows** (x86_64): `"path/to/Reacher_Windows_x86_64/Reacher.exe"`
- **Linux** (x86): `"path/to/Reacher_Linux/Reacher.x86"`
- **Linux** (x86_64): `"path/to/Reacher_Linux/Reacher.x86_64"`
- **Linux** (x86, headless): `"path/to/Reacher_Linux_NoVis/Reacher.x86"`
- **Linux** (x86_64, headless): `"path/to/Reacher_Linux_NoVis/Reacher.x86_64"`

For instance, if you are using a Mac, then you downloaded `Reacher.app`.  If this file is in the same folder as the notebook, then the line below should appear as follows:
```
env = UnityEnvironment(file_name="Reacher.app")
```


```python
env = UnityEnvironment(file_name='./Reacher_Linux/Reacher.x86_64',
                       no_graphics = False)
# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
```

## 3. Instructions: the DDPG algorithm
### 3.1 The DDPG

The deep deterministic policy gradient (DDPG) method [1] is a model free reinforcement learning algorithm, and it is an extension of the deterministic policy gradient (DPG) method [2]. The difference between the two method is that, DPG considers the deterministic policies which considers that

<img src="https://latex.codecogs.com/gif.latex?a=\mu_\theta(s)"/>

where a is the action, <img src="https://latex.codecogs.com/gif.latex?\mu"/> is the policy, <img src="https://latex.codecogs.com/gif.latex?\theta"/> is the parameters and s is the state.

The DDPG method adopts the actor-critic approach with Deep Q Network[3] to form a model-free, off-policy reinforcement learning algorithm for the learning of optimal policies in high-dimensional and continuous action spaces problems, such as autonomous driving and robotics, etc. For the example problems, their actuators receives continuous command, such as throttle and joint torques. The DQN method can only handle discrete action space, for that reason, its application is limited.

### 3.2 The actor-critic

The DDPG uses stochastic policy for the agent, i.e.

<img src="https://latex.codecogs.com/gif.latex?\pi_{\theta}(a|s)=\mathbb{P}[a|s,\theta]"/>

where <img src="https://latex.codecogs.com/gif.latex?\theta"/> is the parameter vector, <img src="https://latex.codecogs.com/gif.latex?\pi"/> is the policy.

For this problem, the stochastic actor-critic method is applied. the actor is applied to find the optimal <img src="https://latex.codecogs.com/gif.latex?\theta^*"/> in order to approach the optimal policy <img src="https://latex.codecogs.com/gif.latex?\pi^*"/>, that's to say, <img src="https://latex.codecogs.com/gif.latex?\pi_{\theta}(a|s)\rightarrow\pi^*_{\theta}(a|s)"/>. For policy gradient method, the state-value function has to be estimated as well. In this approach, the critic is applied to adjust the parameter vector to approximate the sate-value function <img src="https://latex.codecogs.com/gif.latex?Q^{\pi}(s,a)"/>. Then, an approach similar to DQN method is applied for both actor-critic networks.

A thematic diagram for this approach is shown in Fig 1 and Fig 2.  

![actor_learn](https://github.com/FlyienSHaDOw/project_2_continuous_control/blob/master/img/actor_learn.png)

<center>Figure 1 DDPG learning process—actor</center>

For the process of training actor network, the training data are randomly picked from the **Experience Replay Buffer**. The predicted action <img src="https://latex.codecogs.com/gif.latex?a_p\{t\}"/> is generated via **Local** actor network fed by current state <img src="https://latex.codecogs.com/gif.latex?s_t"/>. Then, an approximated action-value function <img src="https://latex.codecogs.com/gif.latex?Q^\omega(s_t,a_{p\{t\}})"/>. An unary minus of the approximated action-value function is directly used as the loss function for the update of the **Local** actor network. 

![critic_learn](https://github.com/FlyienSHaDOw/project_2_continuous_control/blob/master/img/critic_learn.png)

<center>Figure 2 DDPG learning process—critic</center>

 The update of critic network is even more complex. First of all, since we prefer to get the **Expected** action-value function, we use the state of the next time step <img src="https://latex.codecogs.com/gif.latex?s_{t+1}"/>. An next time step action is guessed via the **Target** network of the actor. And The expected value function is generated via **Target** network of the critic, and the action value function is generated via **Local** network of critic. Then, the Bellman equation is calculated with the value function, and the mean-square-error loss function is applied for the update of **Local** network of the critic.

Bear in mind that, for both actor and critic network, the Target network are slowly converged to the Local network through **soft update**.

The **ReplayBuffer** class is a container which stores the past experiences. In the learn procedure, the past experiences are stochastically chosen and are fed into the two Q-networks. One Q-network is fixed as Q-target, it is denoted by<img src="https://latex.codecogs.com/gif.latex?\theta^-"/>. This Q-network is 'detached' in the training process, in order to achieve better stability. As a consequence, the change in weights can be expressed as 

<img src="https://latex.codecogs.com/gif.latex?\Delta\theta=\alpha\left[(R+\gamma\max_{a}\hat{Q}(s,a,\theta^-)-\hat{Q}(s,a,\theta))\nabla_{\theta}\hat{Q}(s,a,\theta)\right]"/>

DDPG is an off-policy algorithm, as a matter of fact, the exploration procedure can be conducted independently. This procedure is kind of policy gradient method. An stochastic actor is determined by the current policy, and noise generated by the **Uhlenbeck & Ornstein** method is added to it for searching the gradient direction, until it approaches the optimal policy. Thus the actor policy can be expressed as

<img src="https://latex.codecogs.com/gif.latex?\pi'(s_t)=\pi(s_t|\theta_t^\pi)+\mathcal{N}"/>

where <img src="https://latex.codecogs.com/gif.latex?\mathcal{N}"/> is the noise for searching 'best' actions.

### 3.2 The model

In this project, the Q-net is constructed by **three fully connected layers**. The architecture is the same as the network described in the paper [1]. But the units are reduced to reduce the computational time, since the problem is simpler.  In this case, the hidden layers are with 128 and 256 units respectively. For the input layer, the number of input node is the same as the number of states of the agent.  Finally, for the output layer, the number of output layer is the same as the action size of the agent. For the input layer and the out put layer, the output value is activated by the **Rectified Linear Unit** (ReLU) function. Since this is a continuous control problem, we have to use **tanh** function for the output of final layer. The network for the critic has the same structure as the actor network. however, the critic approximates the action-value function, its input should be states and action, consequently, the number of node for the input layer is the number of states plus the number of actions. 

## 4. Solution

### 4.1 Hyper-parameters

The hyper parameters for the learning process are generally utilized the parameters provided by  the paper [1]. However, some modifications are conducted for both convergence and stability. The WEIGHT_DECAY is set as 0. And I conduct one training process in very 25 time steps. In my  hyper-parameters tuning experience, TRAIN_EVERY influence the convergence significantly. At one training step, I set NUM_TRAINS as 5 to conduct 5 trains at a time. Other difference  is that I increase the minibatch size to 128 to allow more past experiences to be used for one training. Another improvement is that, I reduce the exploration noise a decay rate (say 0.999) to achieve better stability.


```python
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0.     # L2 weight decay
TRAIN_EVERY = 25        # how often to update the network
NUM_AGENTS = num_agents
NUM_TRAINS = 5
```

### 4.2 The Training Process

At one time step of an episode, the process is generally depicted in Figure 3.  The agent choose an action corresponding to  the current state via the Local network of the actor. And the action is applied to the environment, generates the reward of the action and the state, and the transmission of the next state. Than, they are stored in the Experience Replay Buffer for the training process.

![process_learn](https://github.com/FlyienSHaDOw/project_2_continuous_control/blob/master/img/process_learn.png)

<center>Figure 3 The training process</center>

## 5 Result and Conclusion

### 5.1 Result

The animation shown in Figure 4 demonstrates the effectiveness of the trained network, and the Figure 5 shows the learning procedure. With the prescribed structure and hyper parameters, the networks converges to the 'optimal policy' nicely with little oscillations. And the agent reaches the target average score 30 in 150 episodes, which means the network structure and the hyper parameters defined find a good balance point between exploration and exploitation.

![result_animation](https://github.com/FlyienSHaDOw/project_2_continuous_control/blob/master/img/result_animation.gif)

<center> Figure 4  Performance of the trained agent </center>

![result](https://github.com/FlyienSHaDOw/project_2_continuous_control/blob/master/img/result.png)

<center> Figure 5 The training process </center>

### 5.2 Conclusion

In the parameter tuning process, the author found that the DDPG method is not robust enough, and the tuning process can be painful, since the DDPG is too sensitive for the hyper parameters, but the window for a good value of hyper parameter is too narrow. As a meter of fact, a robust method can be applied in the future work.

## 6 Reference
[1] Lillicrap, T. Hunt, J. Pritzel, A. Heess, N. Erez, T. Tassa, Y. Silver, D. & Wierstra, D. (2016). Continuous Control with Reinforcement Learning, In Proceedings of ICLR. https://arxiv.org/abs/1509.02971

[2] Silver, D., Lever, G., Heess, N., Degris, T., Wierstra, D., & Riedmiller, M.A. (2014). Deterministic Policy Gradient Algorithms. ICML. https://dl.acm.org/doi/10.5555/3044805.3044850

[3] Watkins, C.J., Dayan, P. Technical Note: Q-Learning. Machine Learning 8, 279–292
(1992). https://doi.org/10.1023/A:102267672231

[4] Sutton R, Barto A, Reinforcement Learning: An Introduction, The MIT Press, 2018.
