# Group Members :


* Fauzaan Qureshi - 2017A2PS0663P
* Rohit Bohra - 2017A7PS0225P
* Kushaghra Raina - 2017A7PS0161P

# Atari-Deep-Reinforcement-Learning-

This Repository is part of our NNFL Course Project(Paper ID - 83). 

We have implemented the algorithm(Deep Q-learning with Experience Replay) given in this novel paper.
Paper Link - https://arxiv.org/pdf/1312.5602.pdf

We are currently in the process of training our games. Please refer these drive links to get our latest weights.

AtariSpaceInvaders: https://drive.google.com/drive/folders/1mT960D_pYUN5OaqYBTA4Pjemh10YbQCQ?usp=sharing

AtariBreakout : https://drive.google.com/drive/folders/1bDPZg1KYmhIQ3DThhxLvTkOLnwMzgFpK?usp=sharing


# Deep Q-learning with Experience Replay :
We can't just use a simple Q-table for training games like chess,mario or breakout unless you have a memory size which can handle a state-space of order 10^100 and a super computer to speed up your computations. To avoid this problem, we use a Deep Q network(DQN). We approximate the Q(s, a) value.

To approximate the next Q value, required to train the policy network, we make a clone of the policy network called the Target Network. To this target network, we pass the state of game that is reached due to the action dictated by the policy network's output, i.e, the Q values. Taking this state as input, the target network generates the Q-values for next action.

![DQN](https://github.com/FauzaanQureshi/deep-Q-learning/blob/master/Assets/Others/DQN_Algorithm.png)

![CNN](https://github.com/FauzaanQureshi/deep-Q-learning/blob/master/Assets/Others/CNN.png)

With both experience replay and the target network, we have a more stable input and output to train the network and behaves more like supervised training.

This is the 4 layer CNN architecture used in the paper. We have used the same architecture to train our agent.

# Implementation Details :
We have used Keras for all implementations and Matplotlib to visualize the graphs for rewards and losses. We have implemented the same alogrithm mentioned above. We had previously trained using both optimizers - RMSProp and Adam. Adam worked better for our implementation. Replay Memory Size was fixed at 40000 experiences because of ram limitations. Before training, initial exploration is done to gain some random experiences and fill up our replay memory. Our implementation can be used to train any atari game. We trained **Breakout** and **SpaceInvaders** in a deterministic enviroment using OpenAI Gym library. After training for _ episodes, we started getting satisfying results. Training for SpaceInvaders is still in progress but we hope to complete it soon with good results.

# Results from our experiments :

![Breakout avg](https://github.com/FauzaanQureshi/deep-Q-learning/blob/master/Assets/Results/Breakout%2050%20Avg.png)

![Breakout score](https://github.com/FauzaanQureshi/deep-Q-learning/blob/master/Assets/Results/Breakout%20Score.png)

![SpaceInvaders avg](https://github.com/FauzaanQureshi/deep-Q-learning/blob/master/Assets/Results/SpaceInvaders%20Avg%2010.png)

![SpaceInvaders score](https://github.com/FauzaanQureshi/deep-Q-learning/blob/master/Assets/Results/SpaceInvaders%20Score.png)

# Common Issues faced by us :

* We need to take care of some issues in games like Breakout where fire has to be pressed manually after losing a life. This is done because the Q-value of "fire" becomes very less and it is very hard to determine those 5 occurences(5 Lives) of pressing fire. There is a similar problem in Pong as well.

* Initially in Breakout, our agent was getting decent score but it was because our agent was getting stuck at local minima. The Slider moved to the extreme of the frame to gain some initial advantage but was getting stuck there.

# Instructions to Run :

* pip install -r requirements.txt
* Before training, make sure you set the [hyperparameters](https://github.com/FauzaanQureshi/deep-Q-learning/blob/master/hyperparameters.py) correctly.
* python main.py GAME_NAME (By default, breakout is loaded and games are loaded in a deterministic enviroment(v4))
* Choose among the options : 1 for train, 2 for test, 3 for seeing results on our pretrained weights(Log files are used for this purpose).
* When training for first time, initial exploration will be done first and then training. User has to manually enter the number of episodes he want to train the agent.
* All weights,experiences,logs and results can be found in the Assets folder.
