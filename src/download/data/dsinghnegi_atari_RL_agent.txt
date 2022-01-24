# Atari RL agent

Reinforcement learning based agents to playing Atari games.

## Agents

- Deep Q-network(DQN)
    - Double DQN
    - Prioritized Replay
    - Dueiling network

- Asynchronous Advantage Actor-Critic(A3C)
    - Multiprocess support
    - Multiprocess Cuda support
    - LSTM based model
    - Generalized Advantage Estimate (GAE)
    - Frame stacking based model

## Game Play

### DQN

![Pong](images/pong_low.gif)
![BreakoutDeterministic](images/breakout_low.gif)

### A3C

![KungFuMaster](images/KungFuMaster_low.gif)
![Boxing](images/Boxing_low.gif)
![SpaceInvaders](images/SpaceInvaders_low.gif)

## Supported Environment

- [x] BreakoutDeterministic-v4
- [x] PongDeterministic-v4
- [x] KungFuMasterDeterministic-v4
- [x] BoxingDeterministic-v4
- [ ] SapecInvadersDeterministic-v4

## TODO

- [x] DQN
- [x] TensorBoard support
- [x] Double DQN
- [x] Prioritized replay
- [x] Dueling network
- [X] Train model for Pong
- [x] Achive 300+ score on breakout
- [x] A3C Agent for KungFuMasterDeterministic-v4
- [x] A3C Agent for BoxingDeterministic-v4
- [x] Parallel processing for A3C
- [x] LSTM layer for A3C to replace frame stacking

## License

The Apache-2.0 License. Please see the [license file](LICENSE) for more information.

## References

- https://github.com/yandexdataschool/Practical_RL
- https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
- https://towardsdatascience.com/tutorial-double-deep-q-learning-with-dueling-network-architectures-4c1b3fb7f756
- https://pytorch.org/docs/stable/notes/multiprocessing.html
- https://github.com/ikostrikov/pytorch-a3c
- https://arxiv.org/pdf/1506.02438.pdf
- https://arxiv.org/abs/1602.01783
