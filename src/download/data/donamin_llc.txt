# Learning Task-Agnostic Action Spaces for Movement Optimization

This repository contains the source code for the algorithm, described in [this paper](https://arxiv.org/abs/2009.10337).

![Image: Overview of the system pipeline](overview-diagram.png)

## Abstract
We propose a novel method for exploring the dynamics of physically based animated characters, and learning a task-agnostic action space that makes movement optimization easier. Like several previous papers, we parameterize actions as target states, and learn a short-horizon goal-conditioned low-level control policy that drives the agent's state towards the targets. Our novel contribution is that with our exploration data, we are able to learn the low-level policy in a generic manner and without any reference movement data. Trained once for each agent or simulation environment, the policy improves the efficiency of optimizing both trajectories and high-level policies across multiple tasks and optimization algorithms. We also contribute novel visualizations that show how using target states as actions makes optimized trajectories more robust to disturbances; this manifests as wider optima that are easy to find. Due to its simplicity and generality, our proposed approach should provide a building block that can improve a large variety of movement optimization methods and applications.

## Prerequisites
- Python 3.5 or above
- cma
- glfw
- gym
- Keras
- mujoco-py
- numpy
- opencv-python
- pandas
- Pillow
- stable-baselines
- tensorflow

More detailed requirements are specified in ```requirements.txt```.

## Code Structure

### Primary scripts
- ```NaiveExplorer.py```	The script for generating the exploration data using naive exploration
- ```ContactExplorer.py```	The script for generating the exploration data using the proposed contact-based exploration algorithm
- ```produce_llcs.py```	The script for training the LLCs using the exploration data
- ```offline_trajectory_optimization.py```	The script for offline trajectory optimization using [CMA-ES](https://link.springer.com/chapter/10.1007/3-540-32494-1_4)
- ```online_trajectory_optimization.py```	The script for online trajectory optimization using a simplified version of [Fixed-Depth Informed MCTS (FDI-MCTS)](https://ieeexplore.ieee.org/document/8401544/)
- ```RL_Trainer.py```	The script for reinforcement learning using [PPO](https://arxiv.org/abs/1707.06347) or [SAC](https://arxiv.org/abs/1801.01290)
- ```RL_Renderer.py```	The script for rendering policies trained using [PPO](https://arxiv.org/abs/1707.06347) or [SAC](https://arxiv.org/abs/1801.01290)

### Secondary scripts 
- ```LLC.py```	The script for implementing and training state-reaching LLCs
- ```MLP.py```	Neural network helper class
- ```logger.py```	The logger script, taken from <a href="https://github.com/openai/baselines">OpenAI Baselines repository</a>
- ```RenderTimer.py```	Helper script for helping with realtime rendering

### Data and models (used in the paper)
- ```ExplorationData```	The folder containing the exploration data generated using naive and contact-based exploration methods.
- ```Models```	The folder containing all the LLCs for two exploration methods, four agents, and five horizon values (both in multi-target and single-target mode)
