

#### Preliminary remark: 
The "Navigation" Project is part of the "Deep Reinforcement Learning" Nanodegree study of Udacity.
To get the Nanodegree-Diploma, students have to realize several software projects in a given time.


#### What is Reinforcement Learning ?
A Reinforcement-Learning-System is a System constructed with artificial intelligence technique.
In the following picture you can see the  principle structure of a "Reinforcement Learning"-System:
![GitHub Logo](attachments/ReinforcementLearningPrinciple.JPG)
Such a system is composed of the following componends:

* the agent 
* the environment
* the reward
* the state
* the actions

An agent has to solve a special task inside of a given environment.
The task can be e.g. to find objects in a special environment.
We can think that an Agent is a robot and if he is trained, he can take decisions
on his own and often finds better and faster solutions as human beings.

At the beginning to solve his task, the agent knows nothing about the environment.
The agent just gets  a feed-back from the environment about his proceeded action.
Actions can be e.g. moving forward, backward, left and right, as used in this project. 
One move is one action and the agent gets back an information about the state and he 
gets back also a reward.
With this information of the environment artificial-intelligence-algorithms can find an optimal behavior
for solving the agents task.
We will have a look in detail on the here applied Deep Q-Learning-algorithm  in the
file repord.md.
In recent the years, there have been some crucial technical breakthroughs in the field of artificial  
intelligence which seemed unthinkable 10 years ago.


**To show two examples of this progress I would like to mention two examples:  **  
1.) The science paper from Hasselt, Guez and Silver:

     https://arxiv.org/pdf/1509.06461.pdf  
In this science work from 2015 the scientists developed an algorithm which was applied on 57-Atari-Plays.
The Agent took the role of a human player and got impressive results mostly far better  then human players.  
The special Deep-Q-Learning-Algorithm (Double-Q-Learning) of this science paper is applied in the given Navigation
project. 

2.) Alpha Zero  (See also Wickipedia-Article:   https://en.wikipedia.org/wiki/AlphaZero)   
Alpha Zero  is a computer program developed by artificial intelligence research company DeepMind to master several games.
This Software was able to learn to play chess and reached a world champion level in 4 hours.
The Program learned only by playing against himselves.

Breakehroughs in Neural Network Structures like CNN, RNN, GAN, RL etc. as well as the improved computational CPU-power played an importand role in this progress.


#### What the Navigation project does ?
The goal of the agent is to collect in his environment as many yellow bananas as possible while avoiding blue bananas.
A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. 

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

* 0 - move forward.
* 1 - move backward.
* 2 - turn left.
* 3 - turn right.  

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.


#### Why the project is useful ?
In this project, we can study in detail the behaviour of an agent with the Deep-Q-Learning algorithm.
Beside  some theoretical knowledge of Neural Networks, one need the practical experience  for a 
better understanding of the stuff. This project is created to get it.
We can learn from tweaking parameters and applying cutting edge optimization-techniques of the Deep Q-Learning Algorithm.

#### How users can get started with the project ?
Step 1:  
First, please follow the instructions of the part dependencies on the Link below.
https://github.com/udacity/deep-reinforcement-learning

Step 2: 
Install the Unity-Banana-Environment for your operating system.  

https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation

Step 3:   
Copy the files of my Github-Project-Directory in your local project-directory.

Step 4:  
Start your jupyter Notebook and call the "navigation.ipynb"-File in the working directory.
(If jupyter notebook is not available on your computer, install it:    
https://jupyter.readthedocs.io/en/latest/install.html)

good luck ☺️

