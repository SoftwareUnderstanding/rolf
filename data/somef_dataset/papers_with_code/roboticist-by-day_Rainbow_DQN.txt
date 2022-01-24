## Introduction

This repository contains an implementation of the rainbow DQN algorithm put forward by the DeepMind team in the paper '[Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298)'. [1]


<p align="center" >
  <img width="160" height="210" src="media/pong.gif">
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img width="315" height="210" src="media/cartpole.gif">
</p>


## Usage

For those interested in studying the contributions of various rainbow components, the code supports functionlity to perform ablation studies. However, some combinations might not have the helper functions pre-defined. To disable improvements, modify the arguments passed to the rainbow class as necessary and pass the flags that disable the respective components. 

Training data is written to tensorbaord. Currently, training can be restored by providing the model state dict. The code can also be easily extended to restore the optimizer and other epoch dependent data.

### Linux

To clone the repository,

```
git clone https://github.com/roboticist-by-day/rainbow_dqn.git
cd rainbow_dqn
```

To create the python environment using Anaconda package manager,

```
conda create --name <env_name> --file requirements.txt
conda activate <env_name>
```

To test the rainbow algorithm on your machine,

```
python rainbow/example.py --env='CartPole-v1' --runs=3 --episodes=250  --render
```
#### Arguments:
    env      - available environments are 'CartPole-v1' and 'Pong-v0'

    runs     - number of trials to repeat the training (default 3) 

    episodes - number of episodes to run during each trial run (default 250)

    render   - flag to render agent performance after training

**Note:** The 'Pong-v0' environment requires significant memory to store the replay buffer.


## Pre-Trained Models

Pre-trained models are available for the following environments

    - CartPole-v1
    - Pong-v0


## Requirements
    - python 3.5 or greater
    - pytorch
    - openai gym
    - tqdm (optional)
    - tensorboard (optional)
    - matplotlib (optional)


## Description

Deep Q-Learning Network (DQN) [2] is one of the most popular deep reinforcement learning algorithms. It is an off-policy learning algorithm that is highly sample efficient. Over the years, many improvements were proposed to improve the performance of DQN. Of the many extensions available for the DQN algorithm, some popular enhancements were combined by the DeepMind team and presented as the Rainbow DQN algorithm. These imporvements were found to be mostly orthogonal, with each component contributing to various degrees.

The six add-ons to the base DQN algorithm in the Rainbow version are

    1. Double Q-Learning

    2. Prioritized Experience Replay

    3. Dueling Networks

    4. Multi-step (n-step) Learning

    5. Distributional Value Learning

    6. Noisy Networks

### 1. Double Q-learning
DQN introduced the concept of using of two different networks - a policy network for action selection and a target network for bootsrapping values during learning. This helped to improve stability and thereby avoid oscillations. DQN suffered from an over-estimation bias as the target network always bootstraps with the maximum next state-action value. Hence some noise in the target estimation is sufficient to make the algorithm over-optimiistic.

Van Hassalt et. al [3] introduced a method to alleviate this effect by decoupling the argmax selection and state-action value calculation through the use of two different networks. The policy network is used for maximum action value selection and the target network is used for bootstrapping the selected state-action pair values.

### 2. Prioritized Experience Replay
Prioritized experience replay is a crucial part of DQN. It provides stability by breaking correlations in the agent's experience. The transitions expereienced by the agent is stored in a buffer. A mini-batch is sampled from this buffer uniformly and is used at each learning step. Thus older experience, which is likely to be different from the observations encountered during current policy, is also available for learning. This breaks the correlations and provides a rich and diverse experience to learn from.

A suggested by Silver et al. in [1] some experiences have more prospect for potential learning. Better performance can be obtained by prioritizing transitions with greater learning potential. The td-error of the transition is a readily available proxy for the learning potential metric. Schaul et al. [4] proposed a proportial method and a rank based method to achieve this prioritizing. Though the rainbow paper uses the proportional variant, this implementation uses the rank based variant as it is less sensitive to outliers. An effective implementation of the prioritized queue is necessary to avoid excessive computational overhead. The priority queue is implemented here using a Binary Search Tree (BST).

### 3. Dueling Networks
Wang et al. [5] showed that that the performance of DQN can be improved by decoupling the value estimation and advantage estimation. This dueling architecture has a shared layer of networks followed by two separate streams - one for estimating state value and another for the action advantage. These two streams are again combined to compute the state-action values. This allows for better generalization across actions


### 4. MultiStep Learning
Bootstrapping over multiple time steps has always been an effective technique in temporal difference learning [6]. Mnih et al. [7] in their famous A3C paper proposed the idea of using multistep learning in Deep Reinforcement Learning algorithms like DQN. The n-step variants generally lead to faster learning. Most common choices of n are in the range 3-5.

### 5. Distributional Value Learning
Reinforcement learning literature has predominantly focussed on learning an estimate of the expectation of the return. However, Bellemare et al. [8] showed that learning a distribution rather than a fixed return provides better stability particularly when using funtion approximation. Hence here, the action-value function is approximated as a factorized probability distribution.

### 6. Noisy Networks
Achieving a directed exploration strategy is not very straightforward with Neural Networks. The most common exploration policy - epsilon greedy, is quite inefficient as the exploration is random. Fortunato et al. [9] introduced NosiyNets which have a parameteric noise variable embedded into their layers. This induced stochasticity proves to be an effective exploration strategy over the traditional methods. This implementation uses Noisy Linear layers with factorized gaussion noise.

## Citation
1. Matteo Hessel, Joseph Modayil, Hado van Hasselt, Tom Schaul, Georg Ostrovski, Will Dabney, Dan Horgan, Bilal Piot, Mohammad Azar, David Silver. Rainbow: Combining Improvements in Deep Reinforcement Learning. arXiv preprint arXiv:1710.02298, 2017.

2. Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Bellemare,Alex Graves, Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, et al. Human-level control through deep reinforcement learning. Nature, 518(7540):529–533, 2015.

3. Van Hasselt, H.; Guez, A.; and Silver, D. Deep reinforcement learning with double Q-learning. In Proc. of AAAI, 2094–2100, 2016.

4. Schaul, T.; Quan, J.; Antonoglou, I.; and Silver, D. Prioritized experience replay. In Proc. of ICLR, 2015.

5. Wang, Z.; Schaul, T.; Hessel, M.; van Hasselt, H.; Lanctot, M.; and de Freitas, N. 2016. Dueling network architectures for deep reinforcement learning. In Proceedings of The 33rd International Conference on Machine Learning, 1995–2003.

6. Sutton, R. S., and Barto, A. G. Reinforcement Learning: An Introduction - Second Edition. The MIT press, Cambridge MA, 2008.

7. Mnih, V.; Badia, A. P.; Mirza, M.; Graves, A.; Lillicrap, T.; Harley, T.; Silver, D.; and Kavukcuoglu, K. Asynchronous methods for deep reinforcement learning. In International Conference on Machine Learning, 2016.

8. Bellemare, M. G.; Dabney, W.; and Munos, R. A distributional perspective on reinforcement learning. In ICML, 2017.

9. Fortunato, M.; Azar, M. G.; Piot, B.; Menick, J.; Osband, I.; Graves, A.; Mnih, V.; Munos, R.; Hassabis, D.; Pietquin, O.; Blundell, C.; and Legg, S. Noisy networks for exploration. CoRR abs/1706.10295, 2017.


