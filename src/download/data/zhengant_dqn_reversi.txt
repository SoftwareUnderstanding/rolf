# DQN for Reversi
This is an implementation of deep Q-networks (https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) for learning to play Reversi (https://en.wikipedia.org/wiki/Reversi) via self-play. Bells and whistles on top of a vanilla DQN implementation include:
* Experience replay buffer (https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
* Double Q-learning (https://arxiv.org/pdf/1509.06461.pdf)

To run: 
```
python reversi_dqn.py
```

