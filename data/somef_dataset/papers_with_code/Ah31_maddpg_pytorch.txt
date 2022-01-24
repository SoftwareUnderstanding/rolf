# maddpg_pytorch

This code is a pytorch implementation of MADDPG algorithm presented in the following paper by openai:- https://arxiv.org/pdf/1706.02275.pdf.
The multi-agent environment used here is namely "simple_spread" or cooperative-navigation MAE. The description of various other environments can be 
found here:- https://github.com/openai/multiagent-particle-envs.

**Note**:- The original code which is in tensorflow is run for 60000 episodes while here i have trained for just 10000 episodes because currently I am unable to converge ahead of it. The maximum average reward i am able to reach is around -551 while when i ran theirs on the same environment it's around -416. 
I hope to make it converge better in the near future.

#### **Core training parameters**:-

* BUFFER_SIZE = int(1e6)  # replay buffer size

* BATCH_SIZE = 1024  # minibatch size

* GAMMA = 0.95  # discount factor

* TAU = 0.99  # for soft update of target parameters

* LR_ACTOR & critic= 0.01  # learning rate of the actor

* grad_norm_clipping_actor  & critic= 0.5

* num_units in nn model:- 64

These parameters are the same as used in the main tensorflow code.

### ** Paper citation:-

@article{lowe2017multi,
  title={Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments},
  author={Lowe, Ryan and Wu, Yi and Tamar, Aviv and Harb, Jean and Abbeel, Pieter and Mordatch, Igor},
  journal={Neural Information Processing Systems (NIPS)},
  year={2017}
}
