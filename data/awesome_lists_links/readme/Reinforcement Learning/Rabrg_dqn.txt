# dqn
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rabrg/dqn/blob/main/demo.ipynb)

A PyTorch implementation of DeepMind's DQN algorithm with the Double DQN (DDQN) improvement.

## Usage
```python
import gym
from torch import nn


env = gym.make("LunarLander-v2")
model = nn.Sequential(
    nn.Linear(env.observation_space.shape[0], 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, env.action_space.n),
)

dqn = DQN(env, model)
dqn.learn(n_episodes=300)
```

## Demo
### [Cart Pole](https://gym.openai.com/envs/CartPole-v1/)
![](https://media0.giphy.com/media/R5gyLTlMjHGeCkbBEI/giphy.gif)

### [Lunar Landing](https://gym.openai.com/envs/LunarLander-v2/)
![](https://media0.giphy.com/media/X7O4mXxCdw3s75On50/giphy.gif)

## Related Work
```BibTeX
@article{mnih_playing_2013,
	title = {Playing {Atari} with {Deep} {Reinforcement} {Learning}},
	url = {http://arxiv.org/abs/1312.5602},
	abstract = {We present the first deep learning model to successfully learn control policies directly from high-dimensional sensory input using reinforcement learning. The model is a convolutional neural network, trained with a variant of Q-learning, whose input is raw pixels and whose output is a value function estimating future rewards. We apply our method to seven Atari 2600 games from the Arcade Learning Environment, with no adjustment of the architecture or learning algorithm. We find that it outperforms all previous approaches on six of the games and surpasses a human expert on three of them.},
	urldate = {2021-08-21},
	journal = {arXiv:1312.5602 [cs]},
	author = {Mnih, Volodymyr and Kavukcuoglu, Koray and Silver, David and Graves, Alex and Antonoglou, Ioannis and Wierstra, Daan and Riedmiller, Martin},
	month = dec,
	year = {2013},
	note = {arXiv: 1312.5602},
	keywords = {Computer Science - Machine Learning},
	file = {arXiv Fulltext PDF:/Users/ryangr/Zotero/storage/ANV64XSM/Mnih et al. - 2013 - Playing Atari with Deep Reinforcement Learning.pdf:application/pdf;arXiv.org Snapshot:/Users/ryangr/Zotero/storage/WDPY5Y6P/1312.html:text/html},
}
```

```BibTeX
@article{van_hasselt_deep_2015,
	title = {Deep {Reinforcement} {Learning} with {Double} {Q}-learning},
	url = {http://arxiv.org/abs/1509.06461},
	abstract = {The popular Q-learning algorithm is known to overestimate action values under certain conditions. It was not previously known whether, in practice, such overestimations are common, whether they harm performance, and whether they can generally be prevented. In this paper, we answer all these questions affirmatively. In particular, we first show that the recent DQN algorithm, which combines Q-learning with a deep neural network, suffers from substantial overestimations in some games in the Atari 2600 domain. We then show that the idea behind the Double Q-learning algorithm, which was introduced in a tabular setting, can be generalized to work with large-scale function approximation. We propose a specific adaptation to the DQN algorithm and show that the resulting algorithm not only reduces the observed overestimations, as hypothesized, but that this also leads to much better performance on several games.},
	urldate = {2021-08-21},
	journal = {arXiv:1509.06461 [cs]},
	author = {van Hasselt, Hado and Guez, Arthur and Silver, David},
	month = dec,
	year = {2015},
	note = {arXiv: 1509.06461},
	keywords = {Computer Science - Machine Learning},
	file = {arXiv Fulltext PDF:/Users/ryangr/Zotero/storage/Q42ZS9F7/van Hasselt et al. - 2015 - Deep Reinforcement Learning with Double Q-learning.pdf:application/pdf;arXiv.org Snapshot:/Users/ryangr/Zotero/storage/NKNFH2K8/1509.html:text/html},
}
```
