# Playing Atari Pong with Deep Q-Network: 
## Implementation using TF-Agents, se-lecting efficient optimiser, and right replay buffer size.

pong.ipynb implements a deep RL algorithm that utilises Deep Q-Network (DQN) with an experience replay algorithm (Mnih, et al., 2015). This implementation operates directly on raw pixel observations and learns policies to play an Atari game, Pong.

Below visualisation is using trained policy. 

![grab-landing-page](https://github.com/rikluost/RL_DQN_Pong/blob/master/pong.gif)

The research and use of Reinforcement Learning (RL) algorithms have gained interest since the ground-breaking demonstration by DeepMind took place on 2013 (Mnih, et al., 2013). In that demonstration they showed how a deep learning model could learn to play 1970’s Atari 2600 games from the scratch. Not only did it learn to play the games, but its performance matched or surpassed the best human experts.

The jupyter notebook implementation here utilises TF-Agents RL library for Python, which can simulate various envi-ronments, such as Atari games. It is based on TensorFlow and is developed at Google. Settings are similar to (Mnih, et al., 2015), but with improved optimiser, loss function and replay buffer settings.

## Experience Replay Buffer

Experience replay buffer introduces a new hyper-parameter, replay memory buffer size requiring careful tun-ing (Zhang & Sutton, 2017). Because the replay buffer is implemented as fixed-sized circular storage of tran-sitions, the buffer size directly affects the buffer's oldest policy's age. As the policy is updated every four en-vironment steps collected, the oldest policy in the buffer is the number of environment steps divided by four gradient updates (Fedus, et al., 2020). Hence if we run 800,000 iterations, it is pointless to have a buffer size larger than 200,000 when each trajectory is size of four.

In this setting, buffer size of 50000 seems to work the best.

![grab-landing-page](https://github.com/rikluost/RL_DQN_Pong/blob/master/replay.png)

## Optimiser & loss function

There are two common choices for loss and optimiser pairs, one uses Adam-optimiser and the MSE-loss func-tion, and the other uses the RMSProp optimiser with Huber-loss function. For DQN and Pong environment, Adam optimiser with MSE loss function seems to work much better. It converges nicely already after 400,000 iterations to approximately 19 average return while showing reasonably stable behaviour. It also roughly matches (Mnih, et al., 2013) results. The model with RMSProp and Huber loss, on the other hand, takes a long time to learn, and even after 1.6 million iterations, it had only managed to reach a mean return of only approximately 10. 

![grab-landing-page](https://github.com/rikluost/RL_DQN_Pong/blob/master/opti.png)

## References:

- Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., & Riedmiller, M. A. (2013). Playing atari with deep reinforcement learning CoRR, abs/1312.5602. Retrieved from http://arxiv.org/abs/1312.5602
- Geron, A. (2019). Hands-on machine learning with scikit-learn, keras, and tensorflow: Concepts, tools, and techniques to build intelligent systems (2nd ed.). O’Reilly UK Ltd.
- Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction (2nd ed.). Cambridge: MIT Press.
- Lillicrap, T., P., Hunt, J., Pritzel, A.,Heess, N., Erez, T., Tassa, Y., Silver, D., Wierstra, D. (2016 ) Continuous control with deep reinforcement learning. Retrieved from http://arxiv.org/abs/1509.02971 
- Zhang, S., & Sutton, R. S. (2017). A Deeper Look at Experience Replay. Retrieved from https://arxiv.org/abs/1712.01275
- IUBH. (2020). Reinforcement Learning DLMAIRIL01. Erfurt: IUBH.
- Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., . . . Ostrovski, G. (2015). Humanlevel control through deep reinforcement learning. Nature, 518(7540):529–533.
- Luostari R. (2021). Playing Atari Pong with Deep Q-Network: Implementation using TF-Agents, se-lecting efficient optimiser, and right replay buffer size.  
- Fedus, W., Ramachandran, P., Agarwal, R., Bengio, Y., Larochelle, H., Rowland, M., & Dabney, W. (2020). Revisiting Fundamentals of Experience Replay. Proceedings of Machine Learning Research, ISSN: 2640-3498.


The requirements.txt is included. The code is running smoothly on Ubuntu 20.4 with Python 3.7.9. 
