# Teach a Quadcopter How to Fly!

In this project, I have implemented an agent to fly a quadcopter, and then train it using a reinforcement learning algorithm 
DDPG(Deep Deterministic Policy Gradient).  

## Project Dependencies 

This project requires following libraries to be imported. 

	- Numpy
	- Matplotlib
	- Pandas
	- Sys
	- keras
	- collections		
	- random
	- copy

## DDPG Implementation

I have designed Actor/Critic deep neural network model using Keras. Task of agent is to take off vertically in z direction and reach from position 
(0,0,0) to (0,0,10). Agent controls the rotor speed of 4 rotors. By controlling the rotor speed agents learns to take off vertically. 

Quadcoptor environment is simulation provided by udacity. Both state and action space are continous. That's the reason I have chosen DDPG algo for
training agent. 

Atlast I have plot the average reward score for each episode and observered gradual increase in reward. Also I have plotted cumulative rolling mean
of last 100 episode to get sense of how learning curve is changing when reward is averaged over 100 episode. 

## Project Observations:

Getting started was hardest part of the project as I had to fully understand how DDPG algorithm works and get some intuitive sense of how learning is 
taking place with actor, critic and agent. So to get concept clear I had to revisit lesson videos on Policy gradients and Actor/Critic model multiple times. 
After this I had to fully understand Physics simulation model to understand how task can be defined given continous state and action space. 
Understand size/dimensionations of state/action space and how state is getting transformed after stepping with actions. I had to focus much on designing 
the suitable reward function as early several attempts were not so successful. Agent was not learning well and there was sudden drop in score reward.
Yes it was interesting to learn that much of how agent learns is driven by how good reward function is design. 
I concluded that was what mostly driving the agent performance. Now obivously other hyperparameter tunning with network archtitecture do matters and 
I saw how agent changes learning curves when these parameters are changed. For example I observed large oscillation in reward score is reduced with 
reduction in learning rate passed to actor and critic model.

Now with several experimentation with task, reward function, NN archtitecture, hyperparameters tunning, changing activations, regularizer function values 
and batch normalization etc best learning I could get is shown in jupyter notebook plot.

### Below are some of the further improvements I think can be done.

	- More efficient reward function can be designed to optimize learning process. Reward function I used provides huge positive reward for 
	  taking good step but we have not penalized agent much with negative reward when agent takes bad steps.
	- Network oscillations can be further reduced by tunning hyperparamters.
	- Exploration parameters can be changed for random noise function to check its effect on overall learning.
	- Many times agent reaches local optima and keeps oscillating without further improving. Momentum can be added to learning process to get 
	  pass local optima.
	- More experimentation can be done with actor/critic model archtitecture to improve overall training process.

## Further Reference for DDPG algorithm implementation can be found in original paper
https://arxiv.org/pdf/1509.02971.pdf