# Proximal Policy Optimization - with Keras

This is an implementation of the [PPO algorithm](https://github.com/MatteoBrentegani/PPO/tree/master/PPO_DoubleAction). The agent to move in the environment uses both angular and linear speed.
This feature is also used in the [Multi-agent version](https://github.com/MatteoBrentegani/PPO/tree/master/PPO_MultiAgent).


## Summary

The project implement the clipped version of Proximal Policy Optimization Algorithms described here https://arxiv.org/pdf/1707.06347.pdf

Into the config.yaml file are defined some of the hyper parameter used into the various implementation. Those parameters are initilized with the values proposed in the article.


Here the key point:
* Loss function parameters:
  * epsilon = 0.2
  * gamma = 0.99
  * entropy loss = 1e-3
  
* Network size:
  * state_size = 27 (25 laser scan + target heading + target distance)
  * action_size (angular velocity) = 5
  * action_size2 (linear velocity) = 3
  * batch_size = 64
  * output layer = 8 into 2 streams (5 nodes for angular and 3 for linear velocity)
  * lossWeights for the output layer: 
    * 0,5
    * 0.5
    
The values for the loss weights are the result of some test. With an equal weight the success rate is lower. 

### Prerequisites

 * Python 3
 * Tensorflow
 * NumPy, matplotlib, scipy
 * [Keras](https://keras.io/)
 * [Unity](https://unity3d.com/get-unity/download)

```
# create conda environment named "tensorflow"
conda create -n tensorflow pip python=3.6

# activate conda environment
activate tensorflow

# Tensorflow
pip install tensorflow

# Keras
pip install keras
```

### Training

For start the training run the main.py file into anaconda environment:

```
activate tensorflow
python main.c
```

### Future work

The project currently under development involves the use of a neural network shared between the various agents. In addition, each agent has a critical neural network, similar to that used in previous implementations. 

The results found with the various tests were not positive. The agents after several episodes assumed wrong behavior. 
With the [proposed solution](https://github.com/MatteoBrentegani/PPO/tree/master/PPO_CentralizedNN) it was possible to achieve a good level of success, but not consistently.

Arriving at a certain number of episodes, the agents begin to adopt repetitive behaviors. This leads to two possible results:
 * one of the agents manages to constantly reach the goal while the others do not move;
 * agents remain stationary on the spot
