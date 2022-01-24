# baseline_ddpg
baseline DDPG implementation less than 400 lines

After seeing a few sample implementation on DDPG, I have decided to implement a baseline DDPG within a single python script. And this is done in less than 400 lines, including (hopefully) intuitive comments. This implementation is inpired by the original DDPG paper, as well as the following github repos:

Original DDPG paper: https://arxiv.org/pdf/1509.02971.pdf

Git hub repos:

https://github.com/pemami4911/deep-rl/tree/master/ddpg

https://github.com/floodsung/DDPG

https://github.com/openai/baselines

To keep it a short implementation, I have simplify the following:
1. I only implemented a very basic replay buffer, without any clever tricks such as priorities.
2. I didn't implement any model load/save codes, as the training is relatively quick for simple mujoco envs such as InvertedPendulum-v2 and InvertedDoublePendulum-v2.
3. The tensorboard implementation only contains basic functions, I track the test performance every 100 episode as the scores during training are heavily affected by the exploration noise.

A few changes I added with respect to the other basic DDPG implementations:

1. Similar to many other implementations, I found batch norm on critic tend to destabilize training, and layer norm provide a much better improvement, this is also used in openai baseline DDPG implementation.
2. I combine the critic and actor into the same model class, and also created a internal value estimation using current network, and the gradients update for actor can thereby be directly computed.
3. For exploration noise, I use a simple normal N(0,1) scaled by 0.3.

## How to run
./ddpg.py --env \<env of your choose\> --random_seed \<always good to try a few different random seeds\>

For InvertedPendulum-v2, the test score is expected to reach 1000 within 2k episodes, please try a few different random seeds.
