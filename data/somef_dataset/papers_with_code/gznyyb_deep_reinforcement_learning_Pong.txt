# Deep reinforcement learning with pixel features in Atari Pong Game

This project is intended to build up an intelligent agent able to play and win Pong game (https://gym.openai.com/envs/Pong-v0/). This agent was trained under the methods of Neutral Network and Deep Learning.

## Introduction
In the Pong environment, the agent has three related elements: action, reward and state.

Actions: agent takes the action at time t; there are six actions including going up, down, staying put, fire the ball, etc.
Rewards: agent/environment receives/produces reward, when the opponent fails to hit the ball back towards the agent or the agent get 21 points and win.
State: environment updates state St, which is defined by four game frames’ interfaces stacking together - the Pong involves motion of two paddles and one ball, and background features that the agent need to learn at the game.
The network, suggested by https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf, is used to approximate the action values, which consists of three convolutional neural networks followed by two dense layers. In addition to a network used for training, the other network, which is architecture identical with the first one, gets its weights by copying them from the train network periodically during training and is used to compute the action value label. The other network (called the target network by the paper) is set up to avoid instability in training.

The model is trained using the following three frameworks. 

### Simple Deep Q Learning Using Only Train Network
Initially, the model is trained without the action value labels being computed by the target network. They are instead computed by the train network. Therefore, the train network is used for both the model training and label computing. 

### Simple Deep Q Learning Using Target Network
Then a simple deep-Q network model has been established as a baseline model. This time a target network, whose parameter values are periodically copied from the train network, is utilized to compute the labels. 

### Double Q Network
Lastly a double q network model has been tried to compare its performance with that of the baseline model. This time a best action is chosen by using the train network to compute the action values of the next state and finding the maximum action value. Then the target network is used to compute the action value of this “best action,” which is used as the label. 

## Getting Started

The whole project is done in Google's colaboratory environment (https://colab.research.google.com/). 

* run the "Set Up Google Cloud GPU" section first to set up the GPU for faster computation
* run the sequent chunks of code to start training 

## Results


## Built With

* Tensorflow (https://www.tensorflow.org/) - The python package used to build neural networks and the optimization process
* Numpy (http://www.numpy.org/) - The python package used to transform matrices and perform matrix operations
* Matplotlib (https://matplotlib.org/) - The python package used to visualize the final result 
* Gym (https://gym.openai.com/) - The python package used to provide the gaming environment for the reinforcement learning agent to interact with

## Acknowledgments

* https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py 
* https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On/tree/master/Chapter06/lib
* https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On/blob/master/Chapter06/02_dqn_pong.py
* https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
* https://github.com/ageron/handson-ml/blob/master/16_reinforcement_learning.ipynb
* https://arxiv.org/pdf/1509.06461.pdf

