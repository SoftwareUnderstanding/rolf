# Deep Reinforcement Learning for Atari Breakout Game
Replicating Deep RL papers by DeepMind for the Atari Breakout game. Uses the OpenAI gym environment and Keras for Deep Learning models.

![game](./sample.gif)

# Models Implemented
Deep Q-Network (DQN)

Double Deep Q-Network (DDQN)

Dueling Deep Q-Network (Dueling DDQN)

Asynchronous Advantage Actor Critic (A3C)

# Training
To train a Q-Learning model,

``` python DQN.py```

Specify within the code if `double = True` for Double DQN or `Dueling = True` Dueling DQN.

The exact hyperparameters are according to the paper but are all commented within the code.

To train the A3C model, 

``` python A3C.py```

Specify whether `lstm = True` for a final lstm layer.

Training summary will be outputted to Tensorboard. To visualize,

``` tensorboard --logdir /summary  ```

# Evaluation
To evaluate a trained Q-Learning model,

```python DQNEvaluator.py```

Specify the number of games (default `games = 1`) and whether to render (default `True`).

To evaluate a trained A3C model,

```python A3CEvaluator.py```

Specify the number of games (default `games = 1`) and whether to render (default `True`).

# Replicated Papers

Playing Atari with Deep Reinforcement Learning:

https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

Human-level control through deep reinforcement learning:

https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pd

Deep Reinforcement Learning with Double Q-learning:

https://arxiv.org/pdf/1509.06461.pdf

Dueling Network Architectures for Deep Reinforcement Learning:

https://arxiv.org/abs/1511.06581

Asynchronous Methods for Deep Reinforcement Learning:

https://arxiv.org/abs/1602.01783


# Other References

## Helpful Introductory Blogposts
https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26

https://towardsdatascience.com/tutorial-double-deep-q-learning-with-dueling-network-architectures-4c1b3fb7f756

https://medium.freecodecamp.org/an-introduction-to-deep-q-learning-lets-play-doom-54d02d8017d8

## Discounted Reward Calculation for A3C
https://danieltakeshi.github.io/2018/06/28/a2c-a3c/


