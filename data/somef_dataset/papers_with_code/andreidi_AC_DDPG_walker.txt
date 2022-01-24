# DDPG and TD3 unified implementation and benchmark

This a tf/keras implementation of DDPG [https://arxiv.org/abs/1509.02971](https://arxiv.org/abs/1509.02971) and TD3 [https://arxiv.org/abs/1802.09477](https://arxiv.org/abs/1802.09477) - unified in same Agent class. 
The used env is `'BipedalWalker-v2'` [https://gym.openai.com/envs/BipedalWalker-v2/](https://gym.openai.com/envs/BipedalWalker-v2/).
*Important*: the update of the `actor_online` is done via a wrapper than encapsulates the frozen `critic_online` together with the actual `actor_online`. This way the state is actually propagated up to the Q-value and then backwards. This way we only use high-level api.

