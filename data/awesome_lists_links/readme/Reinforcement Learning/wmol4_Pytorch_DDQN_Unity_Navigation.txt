# Pytorch_DDQN_Unity_Navigation
Deep Reinforcement Learning.

![](agent2.gif)

Shown above: First person view of a Reinforcement Learning agent collecting yellow bananas while avoiding blue bananas.

Uses Unity-ML Banana Navigation environment: https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#banana-collector

Written using Python 3 and Pytorch.

## Deep Reinforcement Learning
Uses Double Q-Learning written in Pytorch. For further info on Double Q-Networks (DDQN): https://arxiv.org/pdf/1509.06461.pdf

## The Environment
#### State Space
Uses a state with 37 numeric features derived from ray tracing (rather than pixel inputs).

#### Action Space
4 possible actions (0, 1, 2, 3) corresponding with the moving forward, backward, and rotating left and right.

#### Scoring
+1 for moving into a yellow banana

-1 for moving into a blue banana

0 elsewhere


###### Custom Scoring
+1 for moving into a yellow banana

-1 for moving into a blue banana

-0.03 elsewhere


#### Termination
The game terminates once the agent has performed 300 actions.

## Dependencies
```
copy
numpy
random
sys
torch
unityagents
```

## Solve criteria
The agent has "solved" the environment if it achieves a consecutive 100-game average score of 13 or higher within 1800 games.

## Usage
Extract the Banana_Windows_x86_64 folder.
All code is contained in the ipynb notebook.

#### To train from scratch:
```
DDQN_run().train()
```

![](agent.gif)

If the agent solves the environment, weights are saved (included) as checkpoint.pth

#### To load saved weights and watch a game:
```
DDQN_run().run_saved_model()
```
Note: Must have weights saved as checkpoint.pth.

## Further details
View report.ipynb to view an explanation of the implementation.
