# Deep Reinforcement Learning

This is a walk through my journey of deep reinforcement learning. This will highlight the papers I've read and implemented.

## Deep Q-Networks (DQN)

[1] Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, Martin Riedmiller. **Playing Atari with Deep Reinforcement Learning**.

https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

*The authors expanded upon the concept of a Q-network in reinforcement learning, by introducing a non-linear appromixation with neural networks. They were able to apply a convolution neural network to parse the raw pixels of gameplay in order to outperform human experts.*

My implementation: https://github.com/Brandon-Rozek/DeepRL/blob/master/PoleBalanceKeras.ipynb





[2] Ziyu Wang, Tom Schaul, Matteo Hessel, Hado van Hasselt, Marc Lanctot, Nando de Freitas. **Dueling Network Architectures for Deep Reinforcement Learning**.

https://arxiv.org/abs/1511.06581

*The authors introduced a way to generalize learning across actions in a DQN by modifying the network architecture. The dueling network represents two separate estimators, one for the state function and the other one for the state-dependent action advantage function.*

My implementation: https://github.com/Brandon-Rozek/DeepRL/blob/master/DuelingPoleBalance.ipynb





[3] Tom Schaul, John Quan, Ioannis Antonoglou, David Silver. **Prioritized Experience Replay**.

https://arxiv.org/abs/1511.05952

*Typically in the training of DQN networks, there exists an experience replay buffer where experiences are sampled after a certain time period to train the neural network.  Before this paper, the sampling was done uniformly across all experiences. The authors expand upon that, allowing for more priority or increasing the likeliness of an experience being sampled if the TD error is high.*

My implementation: https://github.com/Brandon-Rozek/DeepRL/blob/master/PrioReplayPoleBalanceKeras.ipynb