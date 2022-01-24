# Pac-Man-Player
A DQN algorithm that plays Pac-Man for the Nintendo Entertainment System.

The DQN algorithm takes inspiration from both the Rainbow and Ape-X algorithms and is implemented using Keras and Tensorflow. Multiple players play Pac-Man simultaneously and input the collected data into a buffer. Several learners sample data from the buffer and use it to udpate the weights and biases of a global neural network. Data I collected from playing the first level of Pac-Man 52 times is also included in the buffer. My collected data is included in the Expert_Data files. Gym-retro is used to play the game. For copyright purposes, the ROM of Pac-Man for the NES is not included.


Link to Rainbow: https://arxiv.org/abs/1710.02298

Link to Ape-X: https://arxiv.org/abs/1805.11593

Gym Retro: https://openai.com/blog/gym-retro/
