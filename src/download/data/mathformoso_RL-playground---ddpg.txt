# RL-playground---ddpg
Implementation of some common RL models in Tensorflow

Own implementation of DDPG.

This version fits only continuous 1D space.
This version uses gym and Tensorflow1.8

Launch the script in command line with flag:
--train True for training
--env "env_name"for specifying gym environment
--path "path_name" for specifing the saving/loading model dir

Quick notes around ddpg:
DDPG takes advantage of Off-Policy learning for continuous control by alternatively:
- learning Q values with a parametrized function by minimizing TD error
- learning the actions maximizing these Q values with another parametrized function by gradient ascent over Q.
