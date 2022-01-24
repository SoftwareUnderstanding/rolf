# Deep Reinforcement Learning with Super Mario Bros. 1-1
Implementing Google's Double DQN and Experience Replay Buffer for training an agent to beat Super Mario Bros 1-1.

The environment to train and run Super Mario Bros. was built by ppaquette (https://github.com/ppaquette/gym-super-mario).
Make sure to have FCEUX 2.2.3 installed with a valid rom of Super Mario Bros.

The data file for the model needs to be downloaded from here (450 mb download): https://drive.google.com/file/d/1n8WrJuhbtbCQ7M3L6FXFEmWMmfKKKZIK/view?usp=sharing. Place the data file in the same folder as the index and meta files.

To run the project, run the test_dqn_smb.py script.

Double DQN: https://arxiv.org/pdf/1509.06461.pdf
Dueling Network Architecture: https://arxiv.org/pdf/1511.06581.pdf
Prioritized Experienced Replay: https://arxiv.org/pdf/1511.05952.pdf

Part of this project was based off of  Berkeley's CS 294-112 Deep Reinforcement Learning Homework 3 for Atari Games: https://github.com/berkeleydeeprlcourse/homework/tree/master/hw3

Video of the agent completing Super Mario Bros. 1-1: https://drive.google.com/file/d/1hGIL3sNw5bDJLnlBjqzdLfCRHt5KP1fd/view?usp=sharing

If you have any questions, please feel free to contact me.
