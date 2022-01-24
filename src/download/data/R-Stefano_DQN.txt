# Playing Atari with Deep Reinforcement Learning

## About
Replicating Google Deepmind's paper "Playing Atari with Deep Reinforcement Learning"

[Full article on the paper here](http://www.stefanorosa.me/topicboard/artificialIntelligence/DQN)

## Dependencies
* Numpy
* Tensorflow
* Matplotlib
* OpenAI Gym
## Getting started
The network architecture is in `DQN.py`. The class `replayMemory.py` stores and manages the transitions created during training. The file `main.py` is used to run the whole program callable using

`python3 main.py`

The network is saved in the folder **myModel** while the tensorboard's file in **results**

## Result
I implemented DQN on the games Pong and Breakout. I first used the hyperparameters given on the Nature paper but the agent was not able to learn any policy better than a random one. 
The agent was outputting the same *q(s,a)* for different states maybe due to neurons died problem.

This can happens when a big gradient value changes the weights linked to the neuron in such a way that the neuron will always output a very negative logit. So, even if the learning rate was given by the paper and the architecture is the same, could happen that a minimum difference in the settings of the architecture such as using a different weights initializer or a different frame preprocessing could make the learning rate 0.00025 not the optimal one. 

In conclusion, I decided to use the hyperparameters used in [this article](https://medium.com/mlreview/speeding-up-dqn-on-pytorch-solving-pong-in-30-minutes-81a1bd2dff55). More precisely:

* Learning Rate: 0.0001
* Target Update Frequency: 1000 training steps
* Initialize Replay Buffer: 10000 transactions
* Epsilon Decay: 100000 steps
* Final Epsilon: 0.01

Moreover, I used Adam Optimizer instead of RMSProp Optimizer which experimentally has given me better results in a shorter period of time. My DQN required 700 episoded which are around 5 hours and 20 minutes to master Pong and more than 3000 episodes which are around 13 hours and 30 minutes to play decently Breakout.

![Result after training](http://demiledge.com/structureFiles/Images/DQN4.png)

## Sources
* [arXiv](https://arxiv.org/abs/1312.5602) by Deepmind
* [Nature paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) by Deepmind
* [Full article about the paper](http://demiledge.com/artificialIntelligence/DQN.php)
