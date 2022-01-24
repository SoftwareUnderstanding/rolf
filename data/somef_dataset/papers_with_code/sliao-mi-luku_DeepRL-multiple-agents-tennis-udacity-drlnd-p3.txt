# AI Tennis Players



- Reinforcement learning environment by Unity ML-Agents
- This repository corresponds to **Project #3** of Udacity's Deep Reinforcement Learning Nanodegree (drlnd)\
  https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893
- Solving the multiple-agent problem using the multi-agent actor-critic method

In this project, I used the multi-agent deep deterministic policy gradient (MADDPG) algorithm to train 2 agents to play tennis with each other. The goal is to hit the ball to the other side as many times as possible, while avoiding hitting the ball to the ground or out of bounds.

The environment is originally from Unity Machine Learning Agents (Unity ML-Agents). For more details and other environments, please visit:\
https://github.com/Unity-Technologies/ml-agents

This project uses the environment provided by Udacity, which is slightly different from the original Unity environment. To run the codes in this repository successfully, Udacity's environment must be used.

[![p3-env-demo.png](https://i.postimg.cc/RCHdJvhv/p3-env-demo.png)](https://postimg.cc/XXndTS9P)\
**(figure)** *The tennis environment by Unity ML-Agents*

## Project details

In the environment, two agents (with blue and red rackets) are hitting a tennis ball back and forth to each other.
The goal is to train both agents so that they make as many volleys as possible.

- **Number of agents**\
There're 2 agents (tennis players), defined as agent 0 and agent 1.

- **States**\
Each agent can observe the state with the dimension = 24. There are 48 state dimensions in total.

- **Actions**\
The action has 4 dimensions (2 for each agent). The action space is continuous, representing the horizontal and vertical movement of the player.

- **Rewards**\
If the agent hits the ball over the net, a reward of +0.1 is provided. If the ball hits the ground or out of bounds, a reward of -0.01 is provided.

- **Goal**\
When an episode ends, we use the maximum of the scores of agent 0 and agent 1 as the score of the episode. The environment is considered solved when the **average of the scores of 100 consecutive episodes** reaches above **+0.5**.


## Getting started

Please follow the steps below to download all the necessary files and dependencies.

1. Install Anaconda (with Python 3.x)\
    https://www.anaconda.com/products/individual
    
2. Create (if you haven't) a new environment with Python 3.6 by typing the following command in the Anaconda Prompt:\
    `conda create --name drlnd python=3.6`
    
3. Install (need only minimal install) `gym` by following the **Installation** section of the OpenAI Gym GitHub:
    https://github.com/openai/gym#id5
    
4. Clone the repository from Udacity's drlnd GitHub
    ``` console
    git clone https://github.com/udacity/deep-reinforcement-learning.git
    cd deep-reinforcement-learning/python
    pip install .
    ```
    (For Windows) If the error "Could not find a version that satisfies the requirement torch==0.4.0 (from unityagents==0.4.0)" occurs, please refer to this thread:\
    https://github.com/udacity/deep-reinforcement-learning/issues/13
  
5. Download the Tennis Environment (Udacity's modified version)\
    Linux: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip \
    Mac OSX: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip \
    Windows (32-bit): https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip \
    Windows (64-bit): https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip
    
    Extract the .zip file and move the folder `Tennis_Windows_x86_64` (or `Tennis`, `Tennis_Linux`, `Tennis_Windows_x86`, depending on the operating system) into the folder `p3_collab-compet` from Step 4.

6. Download all the files (see the table below) from this repository. Place all files in the folder `p3_collab-compet` from Step 4.

    | File Name | Notes |
    | ----------- | ----------- |
    | SL_Tennis_MADDPG.ipynb | main code |
    | maddpgAgent.py | ddpg agent class |
    | networkModels.py | architectures of actor and critic |
    | buffer.py | replay buffer |
    | noiseModels.py | noise process |
    | checkpoint_actor_0.pth | saved weights for Agent 0's actor |
    | checkpoint_critic_0.pth | saved weights for Agent 0's critic |
    | checkpoint_actor_1.pth | saved weights for Agent 1's actor |
    | checkpoint_critic_1.pth | saved weights for Agent 1's critic |

7. You're ready to run the code! Please see the next section.

## How to run the code

Please follow the steps below to train the agents or to watch pre-trained agents perform the task.

#### 1. Run the *Anaconda Prompt* and navigate to the folder `p3_collab-compet`
``` cmd
cd /path/to/the/p3_collab-compet/folder
```
#### 2. Activate the drlnd environment
``` cmd
conda activate drlnd
```
#### 3. Run the Jupter Notebook
``` cmd
jupyter notebook
```
#### 4. Open `SL_Tennis_MADDPG.ipynb` with Jupyter Notebook
#### 5. Run `Box 1` to import packages
Paste the path to `Tennis.exe` after **"file_name = "**

for example, `file_name = "./Tennis_Windows_x86_64/Tennis.exe"`
#### 6. Run `Box 2` to set the hyperparameters
For information of the hyperparameters, please refer to `Report.md`
#### 7. Run `Box 3` to start training the agents
A figure of the noise simulation will be displayed first, which can be used for tuning the parameters of the noise process. Please note that this simulation is independent of the noise process used in the maddpg traning, i.e., the exact noise values generated during training will be different from the values shown in the figure.

After training, the weights and biases of the actor and critic networks will be saved with the file names:\
 `checkpoint_actor_0.pth` and `checkpoint_critic_0.pth` for Agent 0, and\
 `checkpoint_actor_1.pth` and `checkpoint_critic_1.pth` for Agent 1
#### 8. (Optional) Run `Box 4` to load the saved weights into the agents and watch the performance
#### 9. Before closing, simply use the command `env.close()` to close the environment


## References in this project
1. R. Lowe et al., 2017. *Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments*\
https://arxiv.org/abs/1706.02275
2. T. P. Lillicrap et al., 2016. *Continuous control with deep reinforcement learning*\
https://arxiv.org/abs/1509.02971
3. Udacity's GitHub repository **ddpg-pendulum**\
https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum
4. Udacity's drlnd jupyter notebook template of **Project: Collaboration and Competition**
5. Udacity's drlnd MADDPG-Lab (maddpg.py)
