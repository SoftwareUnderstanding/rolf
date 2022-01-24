# CartPole_Optimized_DDQN
This repository contains the implementation of a DDQN agent to solve the CartPole-v0 problem. I use openai_gym to simulate the environment.

The network implemented is a Double Deep Q-learning Network. Using Hindsight Experience Replay to perform online mini-batch learning. The framework used is Tensorflow.
References:
https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
https://arxiv.org/pdf/1509.06461.pdf

This network is implemented as an Agent that interacts with an Environment, which takes an action as input and returns both a reward and an observation as output.

The environment used is the OpenAI Gym CartPole-v0 problem which has a 2-dimensional discrete action space and a continuous 4-dimensional system-state observation space.

Hyperparameter tuning:
To further improve the performance of the AI it has been wrapped in a function which takes hyperparameters as input and returns a quality factor indicating the performance of the AI after every epoch. The current implementation uses an epoch of 300 episodes and returns the average cumulative reward for the last 100 episodes of each epoch.

This function is passed to a Gaussian Process-based Bayesian Optimizer written by Sheffield Machine Learning. Where the hyperparameters are used as the predictors in a black box equation that returns -1 x Average Commulative Reward.  
https://github.com/SheffieldML
