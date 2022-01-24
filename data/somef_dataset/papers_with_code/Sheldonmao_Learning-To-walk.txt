# Learning-To-walk-project

## Motivation
Deep reinforcement learning, which combines deep learning with reinforcement learning, has shown significant effect in decision making problems. However, it requires vast computational resources to achieve a promising performance, for example, even training Atari game using deep Q-learning algorithm requires hundreds of millions of game frames \cite{he2016learning}. In fact, the sparsity and delay character of reward signal serves as an potential reason. (e.g. in the case that an agent can only achieve reward at the end of the game, it takes time for the reward to back-propagate to previous states, thus make it slower)

To address this problem, He et.al. proposed an algorithm using fast reward propagation via optimality tightening that capture rewards within a few steps from both forward and backward steps of replays \cite{he2016learning}. This modified Q-learning algorithm shows efficiency in discrete state space. And We wonder whether the idea of utilizing rewards from "k-neighbours" states will accelerate the deep reinforcement learning problem in continuous state spaces as well (This idea is from CS294 Deep RL Project Suggestions - Fall 2018).

## Problem Statement
By investigating the reward function of Q learning more carefully, we can improve the speed of propagation and achieve improved convergence by longer state-action-reward sequences. In this way, we are able to propagate the rewards directly to steps away, and thus tighten the optimality by finding upper and lower bound using these information. 

However, a significant assumption is based on the discrete space of action and state. Our project intends to extend the method so that it can be applied to the continuous control and advanced performance. For example, by experimenting the actor-critic algorithm, some classic problems such as swing-up, dexterous manipulation, legged locomotion and car driving can be expected. Our hypothesis is that the idea of using longer state-action-reward sequences is also applicable in continuous states and will accelerate the training.

## Proposed Approach
To simulate the walking task, we will use various models built in the OpenAI Gym environment, such as the ant, the cheetah and the humanoid robot. We plan to start from continuous toy models in Gym (e.g. cart-pole) and ultimately apply it in humanoid robot. To apply the method proposed in \cite{he2016learning} to continuous state spaces during walking task, some methods such as actor-critic based Deep Deterministic Policy Gradient (DDPG) \cite{lillicrap2015continuous} and Normalized Advantage Functions (NAF) \cite{gu2016continuous} might be considered.

In this project, we plan to adapt the idea of utilizing the rewards from "k-neighbour" steps to the continuous state space by exploring both DDPG and NAF methods and compare them in detail.

Let's learning to walk in a day!

## Resources:

OpenAI gym: https://gym.openai.com/

Learning to play in a day: https://openreview.net/pdf?id=rJ8Je4clg

DDPG: https://arxiv.org/pdf/1509.02971v5.pdf

NAF: https://arxiv.org/pdf/1603.00748.pdf

source code of “LEARNING TO PLAY IN A DAY: FASTER DEEP REIN- FORCEMENT LEARNING BY OPTIMALITY TIGHTENING”: https://github.com/ShibiHe/Q-Optimality-Tightening

Playing Atari with Deep Reinforcement Learning:https://github.com/spragunr/deep_q_rl
