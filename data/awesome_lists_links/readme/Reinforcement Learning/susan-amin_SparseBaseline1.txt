# SparseReward

This package provides comprehensive tools for examining rl algorithms on sparse and non-sparse environments.

At current version, we only offer Mujoco environment for tests. Current supported algorithms are as follows:

1- https://arxiv.org/pdf/1801.01290.pdf

2- https://arxiv.org/pdf/1706.01905.pdf

3- https://arxiv.org/pdf/1509.02971.pdf

4- ...

In the directory  ``engine/reward_modifier``, we provide different sparsity version for environments.

The typical code to run the program is as follows:

``` python3 main.py```

If you want to change the environment you should type:

``` python3 main.py --env_name Ant-v2```
The environments by defualt are non-sparse. You can make argument sparse by using the following command:

``` python3 main.py --env_name Ant-v2 --sparse_reward --threshold_sparsity 0.05```
where ``--sparse_reward`` makes the environment's reward sparse and ``--threshold_sparsity`` determines the extent of sparsity.

If you want to change the algorithm to run: 

``` python3 main.py --env_name --algo SAC```

Defualt algorithm is DDPG. Current supported algorithms are: ``--algo SAC, --algo DDPG, --algo DDPG_PARAM ``

If you want to change the parameters of specific algorithm (for example SAC) you should write:

``` python3 main.py --env_name --algo SAC --tau_sac 1```

which changes the ``tau `` parameter of SAC algorithm. Detailed information regarding the different setting of algorithms can be found in main.py argparser. We have provided detailed documentation for each parameter.
