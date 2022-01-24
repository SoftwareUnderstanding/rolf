# A2C2: A2C + criticality

This repository investigates the potential of Self-Organized Criticality (SOC) as a method to speed learning, in particular in a reinforcement learning context. Criticality is implemented practically by the addition of another loss term:

![Image of SOC loss term](https://github.com/AI-RG/rl-experiments/blob/master/lsoc.gif),

which penalizes the time-averaged hidden state *s* (element-wise). This penalty encourages the time average of each component of the state to change over the course of the averaging timescale, so that consistently large (near absolute magnitude 1) or small (near zero) time averages are penalized. One perspective on this approach is that it encourages exploration in the space of internal representations. By penalizing frozen components of hidden states, we incentivize models to take fuller advantage of their representational capabilities.

This repository is modified from a version of the A2C algorithm in OpenAI's collection of baselines.

- Original paper: https://arxiv.org/abs/1602.01783
- Baselines blog post: https://blog.openai.com/baselines-acktr-a2c/
- `python -m baselines.a2c.run_atari` runs the algorithm for 40M frames = 10M timesteps on an Atari game. See help (`-h`) for more options.
