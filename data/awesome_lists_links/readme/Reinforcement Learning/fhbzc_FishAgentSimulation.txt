FishAgentSimulation
=======================

This is a project used to simulate human behaviors in a social learning game. ActorCritic Algorithm is used in the code. Reinforcement 
learning algorithms are copied from a Google project. The paper describing that project can be found at:  
http://arxiv.org/pdf/1509.02971v2.pdf  

How to train the network to simulate the agents?
-------

run 

    python RL_Training_Asocial.py

to start the simulation for Asocial Class.  
A folder called "results_asocial" should be automatically created and the paramaters of neural network will be stored in that folder.  

How to generate simulated agents based on trained model?
-------

run

    python RL_Training_Asocial.py -e 1

csv file for simulated agents behaviors will be stored in folder "results_asocial", another folder called "npz_train" will also be automatically created to store "npy" file for simualted agent behaviors, which will be used to train AI classifier.

An introduction about each file
-----

**train_RL.py**  
Contains class *FishEnv*, which defines the environment of the game.  
Contains class *Actor*, *ActorTarget*, *Critic*, *CriticTarget*, which defines the Actor and Critic in the model.  
Contains method *train*, which is the main method of training.  

**replay_buffer.py**  
A buffer used to store state, action, reward at each timestep during the simulation, being sampled in the later stage to train the neural network in reinforcement learning.  

**neural_network_share_weight.py**  
Contains neural network definition.  

**RL_Training_Asocial.py**  
Contains definition of reward function.
