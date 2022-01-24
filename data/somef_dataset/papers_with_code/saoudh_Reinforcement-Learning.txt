# Reinforcement-Learning
## 1 MDP - Markov Decision Process 
A simple Implementation of MDP in Python 

## 2 Q-Learning 
Implementation of the Q-Learning Algorithm 

## 3 DDPG
This implementation is based on the paper CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING of Timothy P. Lillicrap et al., 2015 https://arxiv.org/pdf/1509.02971.pdf

### 3.1 Evaluation
The error of the policy is depicted as actor-error and the error of the value-function as critic-error. Also the value-function and reward (both, during training and evaluation) curve are plotted too.
Parameters: The neural network has 25 neurons and 1 layer and runs over 100 episodes. Batch Normalization is used too for stabilization. 

### 3.2 Performance 
For testing purposes only 25 neurons and 1 layer is used in this experiment and it converges after 60 episodes. With 150 neurons it even converges after less then 10 episodes with this implementation. 

![pendulum performance](https://github.com/saoudh/Reinforcement-Learning/blob/master/DDPG-master/screenshots/pendulum-performance.png)

### 3.3 Build
The algorithm can be executed by running ddpg.py with python3.

```
python3 -m rlrunner.ddpg
```

The requirements are:

```
gym
tensorflow
```
