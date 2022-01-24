# robot-sac
- Repo for CS 8903 Special Problems Course at Georgia Tech.
- Implementation of [Soft Actor-Critic algorithm by Haarnoja et. al.](https://arxiv.org/abs/1801.01290), [Deep Deterministic Policy Gradients by Lillicrap et. al.](https://arxiv.org/pdf/1509.02971.pdf) and [Hindsight Experience Replay by Andrychowicz et. al](https://arxiv.org/pdf/1707.01495.pdf)
- Implemented using Swift for Tensorflow, Tested on Open AI: Gym environments
- [Swift for TensorFlow Repo](https://github.com/tensorflow/swift)
- [Final Presentation for CS 8903 Class](https://docs.google.com/presentation/d/1HyajlIzJO8N1kSTTjGVlhQaWTl2v7JcxpiJmDQG4IOs/edit?usp=sharing)

# Deep Deterministic Policy Gradients
- [DDPG Paper:](https://arxiv.org/pdf/1509.02971.pdf) 
- The implementation of this algorithm can be found in ddpg.swift. This script contains code for the Actor and Critic networks and also includes the training
  setup for the DDPG algorithm.
- To run this script simply run "swift ddpg.swift"  This script will train a DDPG agent on the inverted pendulum problem from [gym](https://gym.openai.com/envs/Pendulum-v0/)
- I also wrote another script to train the DDPG agent on the Bipedal Walker environment. That can be found in ddpg_walker.swift
- I also made a notebook on Google Colab with this same code:[Link to Notebook](https://colab.research.google.com/drive/1Lmf-CVubsPRhPmcfJ3Dc-gpWXVnF3dcZ?usp=sharing)

# Soft Actor Critic 
- [Soft Actor Critic Paper:](https://arxiv.org/pdf/1801.01290.pdf)
- The implementation for this algorithm can be found in sac.swift. This script contains code for the Gaussian Actor as well as implementations for the Q(s, a) network and the V(s) network. The training setup can also be found in this script.
- I based my implementation of of a python implementation of the algorithm found in this repo: https://github.com/keiohta/tf2rl
- To run this script simply run "swift sac.swift" The script will train the SAC agent on the inverted pendulum problem from [gym](https://gym.openai.com/envs/Pendulum-v0/)
- You can also run this code on a Google Colab notebook [Link to Notebook:](https://colab.research.google.com/drive/1ew6UWWDxjtvnj1vygbcTDBSlmKRDm8N9?usp=sharing)

# DDPG + Hindsight Experience Replay
- [Hindsight Experience Replay Paper: ](https://arxiv.org/pdf/1707.01495.pdf)
- The Hindsight Experience Replay Algorithm deals with Sparse Reward environments and has demonstrated great performance on complex robotic control tasks when combined with DDPG.
-  The implementation for this algorithm can be found in ddpg_her.swift. To run the script simply run "swift ddpg_her.swift" from the command line. The script will train the DDPG + HER agent on the Fetch Push problem from [gym](https://gym.openai.com/envs/FetchPush-v1/)
