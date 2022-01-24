# TD3-deep-rl-research

This repo is originally cloned from https://github.com/sfujim/TD3 which implement TD3 Deep Reinforcement algorithm along with a re-implementation of DDPG.

The idea was to try improve the buffers with entropy maximisation of experience selection combined with a prioritized experience replay:

* DDPG article : https://arxiv.org/abs/1509.02971
* TD3 article : https://arxiv.org/abs/1802.09477
* Prioritized experience replay : https://arxiv.org/abs/1511.05952
* Fractal AI : https://arxiv.org/abs/1803.05049

I used the principle of Virtual Reward, which combined entropy maximisation of observation (or states) with temporal difference errors in the buffer. The initial intuition is that it will reduce varience in the training of the agent, because the training step will try to maximize jointly diversity in observation and error in value function prediction.

For DDPG/TD3 with entropy/error maximisation training:
```bash
python train/main_faimemory.py
```

For original TD3/DDP training:
```bash
python train/main.py
```
