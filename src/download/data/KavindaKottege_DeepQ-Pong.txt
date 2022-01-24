# DeepQ-Pong
A reinforcement learning agent that uses Deep Q Learning with Experience Replay to learn how to play Pong. 

This method of artificial intelligence is very general and can be applied to wide variety of models to find an optimal behaviour to achieve a designated goal.

Inspired by 'Playing Atari with Deep Reinforcement Learning' by DeepMind.
https://arxiv.org/pdf/1312.5602v1.pdf

Simply put, the agent works by: 
Observing the screen output given by the PONG game as well as the score.
Based on this it will choose actions, beginning randomly. 
After observing how its actions, given the state of the PONG game is resulting in a change in score, the agent will optmise it actions as to increase its own score and decrease its opponents. 
Repeating this process many times, the agent eventually learns how to cosistently win the game. 

The Q-value function approximater is a convolutional neural network created using the Keras API from TensorFlow.

After playing 1115 games, the agent won its first game.
After playing 1300 games, the agent won more than 50% of all games.
After playing 2000 games, the agent won 96% of all games. With an average score of 21 against 4.
