# Project_RL

This is a group project for the Reinforcement learning course offered in M.Sc Artificial Intelligence 
at the University of Amsterdam.

### Contributors
* [David Speck](https://github.com/Saduras)
* [Masoumeh Bakhtiariziabari](https://github.com/mbakhtiariz)
* [Ruth Wijma](https://github.com/rwq)
* [Victor Zuanazzi](https://github.com/VictorZuanazzi)

## Dependencies

This project uses python 3. To install dependencies run:
```
pip install -r requirements.txt
```

## Problem Statement

There are different types of experience replay, e.g. prioritized experience replay and hindsight experience replay. Compare two or more types of experience replay. Does the ‘winner’ depend on the type of environment?

## Experience Replays
We mainly experimented with three experience replays techniques which are:
```
- Naive Experience Replay 
- Prioritized Experience Replay
    + rank base
    + proportion base
- Combined Experience Replay
- Adaptive Experience Replay 
  + Adaptive ER
  + Adaptive CER
  + Adaptive rank PER

```

## Hyperparameters
hyperparameters were chosen based on referenced papers 


| Parameter  | Value |
| ------------- | ------------- |
| Fixed Buffer size (Number of frames) [[1](https://arxiv.org/pdf/1712.01275.pdf)]  | 10^4 |
| Batch Size  | 64 |
| Learning rate [[1](https://arxiv.org/pdf/1712.01275.pdf)] | 5e-4 |
| max_step (Maximum episode duration) [[1](https://arxiv.org/pdf/1712.01275.pdf)]  | 1000 |
| ![img](http://latex.codecogs.com/svg.latex?%5Calpha%0D%0A) (Priority) [[2](https://arxiv.org/pdf/1511.05952.pdf)]  | 0.6 |
| ![img](http://latex.codecogs.com/svg.latex?%5Cbeta)  (IS) [[2](https://arxiv.org/pdf/1511.05952.pdf)]  | 0.4 -> 1 |
| ![img](https://latex.codecogs.com/gif.latex?%5Cgamma)  (Discount Factor)  | <ul><li>0.8 for Catpole</li><li>0.99 for MountainCar and LunarLander</li></ul> |
| ![img](https://latex.codecogs.com/gif.latex?%5Ctau)  (Softupdate interpolation parameter)  | <ul><li>0.001 for Catpole</li><li>0.1 for MountainCar and LunarLander</li></ul> |







* [Code](code/)
* [Poster presentation](Poster.pdf)

