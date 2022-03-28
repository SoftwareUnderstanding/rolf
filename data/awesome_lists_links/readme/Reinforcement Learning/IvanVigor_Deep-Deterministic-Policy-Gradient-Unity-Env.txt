# Deep Deterministic Policy Gradient for Reacher Unity Env

This repository contains all the tools and code for running a Deep Deterministic Policy Gradient (DDPG)  network over a [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) Unity environment. 

![Alt Text](https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif)

In this environment, a double-jointed arm can move to target locations. The double-jointed arm receive a reward of +0.1 for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

For this environment we used only single double-jointed arm. It has been studied that using multiple arms may reduce the training effort thanks to the [sharing of experience](https://ai.googleblog.com/2016/10/how-robots-can-acquire-new-skills-from.html). For this project the task can be considered as completed when the average of last 100 episodes is greater than 30.


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

- **`report.md`**: it is a report file
- **`ddpg_agent.py`**: this file contains the Agent and Critic learning approach
- **`model.py`**: topology of the two networks
- **`Continous_control.ipynb`**: Notebook related to the environment

## References

Deep Deterministic Policy Gradient - https://arxiv.org/abs/1509.02971
Reacher Challenge - https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher

## Author

Ivan Vigorito

## License

GNU GENERAL PUBLIC LICENSE [Version 3, 29 June 2007]
