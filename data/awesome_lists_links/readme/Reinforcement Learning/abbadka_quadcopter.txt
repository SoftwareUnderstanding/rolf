# Quadcopter Project

This project is an exercise in reinforcement learning as part of the **Machine Learning Engineer Nanodegree** from Udacity. The idea behind this project is to teach a simulated quadcopter how to perform some activities. I have chosen to teach it two tasks: Take off and maintain position (or hover).

## Concept

The agent is based on the theory in the Deep Deterministic Policy Gradient (DDPG), specifically the concepts from this paper: https://arxiv.org/abs/1509.02971. This method is a special variant of Actor-Critic learning.

## Running the code

To run the code, you can use anaconda. Create an environment as follows:

```
conda create --name quadcopter python=3
```

To use the environment, execute the following:

#### For mac / linux:

```
source activate quadcopter
```

#### For windows:

```
activate quadcopter
```

Afterwards, install the requirements as follows:

```
conda install numpy matplotlib jupyter notebook
```

### Notes

The base code provided for this project is taken from the Nanodegree sessions.
