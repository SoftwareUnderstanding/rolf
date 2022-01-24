# MADDPG-Unity-Env

In this project, I adopted a Multi-Agent Deep Deterministic Policy Gradien for creating two agents with are in charge of collaborate and compete for playing a tennis match. The environment is the similar to the Unity Tennis one. 

![Image](https://www.katnoria.com/static/tennis_play.debc77e3.gif)

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.

##  How to Start

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6.

    - __Linux__ or __Mac__: 
    ```bash
    conda create --name drlnd python=3.6
    source activate drlnd
    ```
    - __Windows__: 
    ```bash
    conda create --name drlnd python=3.6 
    activate drlnd
    ```
    
2. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.  
    - Next, install the **classic control** environment group by following the instructions [here](https://github.com/openai/gym#classic-control).
    - Then, install the **box2d** environment group by following the instructions [here](https://github.com/openai/gym#box2d).
    
3. Clone the repository (if you haven't already!), and navigate to the `python/` folder.  Then, install several dependencies.

```bash
git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
pip install .
```
4. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.  
```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

5. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu. 

##  PyTorch

The model has been developed using PyTorch library. The Pytorch library is available over the main page: https://pytorch.org/

Through the usage of Anaconda, you can download directly the pytorch and torchvision library. 

```bash
conda install pytorch torchvision -c pytorch
```

## Additional Libraries

In addition to PyTorch, in this repository has been used also Numpy. Numpy is already installed in Anaconda, otherwise you can use:

- **`UnityEnvironment`** 
- **`PyTorch`** 
- **`Numpy`** 
- **`Pandas`**
- **`Time`**
- **`Itertools`**
- **`Pandas`**
- **`Time`**
- **`Matplotlib`**

## Files inside repository

- **`report.md`**: Report File
- **`scripts/model.py`**: topology of PyTorch networks
- **`scripts/ddpg_agent.py`**: Agent topology
- **`Tennis.ipynb`**: Contains the Jupyter notebook for running the experiments
- **`agen_weights_X.pth`**  : Actor weights for the Agent number X (there are 2 agents)
- **`critic_weights_X.pth`**  : Critic weights for the Agent number X (there are 2 agents)

## References

Deep Deterministic Policy Gradient - https://arxiv.org/abs/1509.02971
Reacher Challenge - https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher

## Author

Ivan Vigorito

##  License
The code is provided with MIT license 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
