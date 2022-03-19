## Description 

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

![Trained Agent][image1]

The RL agent is allowed to traverse across a two dimensional grid with blue and yellow bananas placed across it. The agent is expected to collect the yellow bananas while avoiding the blue ones. The agent receives a positive reward for every yellow banana it collects and a negative reward for every blue banana collected. The size of the state space is 37. The agent is able to move forwards and backwards as well as turn left and right, thus the size of the action space is 4. The minimal expected performance of the agent after training is a score of +13 over 100 consecutive episodes.

## Algorithm Used

The current solution implements a Dueling DQN algorithm with Prioritized Experience Replay as described in the "Dueling Network Architectures for Deep Reinforcement Learning" paper [(arXiv:1511.06581)](https://arxiv.org/abs/1511.06581)

The network's architecture looks like :
```
DuelQNet(
  (input): Linear(in_features=37, out_features=100, bias=True)
  (dropout): Dropout(p=0.3)
  (hidden): ModuleList(
    (0): Linear(in_features=100, out_features=64, bias=True)
    (1): Linear(in_features=64, out_features=64, bias=True)
    (2): Linear(in_features=64, out_features=64, bias=True)
    (3): Linear(in_features=64, out_features=64, bias=True)
  )
  (value): ModuleList(
    (0): Linear(in_features=64, out_features=64, bias=True)
    (1): Linear(in_features=64, out_features=1, bias=True)
  )
  (advantage): ModuleList(
    (0): Linear(in_features=64, out_features=64, bias=True)
    (1): Linear(in_features=64, out_features=4, bias=True)
  )
)
```

The hyperparameters selected for the demonstration are:
* Learning Rate: 0.0005
* Batch size : 64
* Learns after every steps : 4
* Gamma : 0.99
* Tau : 0.003

It took the agent about 471 episodes to be able to perform with not less than score of 13 as an average of 100 episodes. 

## Plot of Rewards

![](https://github.com/prajwalgatti/DRL-Navigation/raw/master/plot.png)

The goal set of the agent is to reach +13 points as average reward of the last 100 episodes.
The current solution manages to reach the goal after 400-500 episodes and keep improving over 17 points.

The saved weights can be found [here.](https://github.com/prajwalgatti/DRL-Navigation/tree/master/saved_weights)

Follow the training notebook [here.](https://github.com/prajwalgatti/DRL-Navigation/blob/master/Navigation.ipynb)

## Some ideas for future work

* Search for better hyperparameters of algorithm as well as neural network
* Implement prioritized experience replay mechanism
