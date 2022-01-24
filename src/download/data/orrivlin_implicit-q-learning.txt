# implicit-q-learning

A simple PyTorch implementation of the algorithm from the paper: "Offline Reinforcement Learning with Implicit Q-Learning", by Kostrikov et al: https://arxiv.org/abs/2110.06169.
The code borrows heavily from Scott Fujimoto's [original implementation of TD3+BC](https://github.com/sfujim/TD3_BC), with some code borrowed from [d3rlpy](https://github.com/takuseno/d3rlpy/tree/ae983d0f0e4b83c545640b8ffc15f466becef817).
I compared my implementation with the [original JAX source code for IQL](https://github.com/ikostrikov/implicit_q_learning) on the [D4RL-PyBullet datasets](https://github.com/takuseno/d4rl-pybullet) and got similar results, so I hope it is correct.
