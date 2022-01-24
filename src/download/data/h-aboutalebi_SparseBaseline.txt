# SparseReward

This package provides comprehensive tools for examining rl algorithms on sparse and non-sparse environments.

At current version, we only provide the implementation of algorithms in Mujoco environments. 
The supported algorithms are as follows:

1- SAC: https://arxiv.org/pdf/1801.01290.pdf

2- DDPG_PARAM: https://arxiv.org/pdf/1706.01905.pdf

3- DDPG (with uncorrelated noise) / DDPG_NO_NOISE / DDPG_OU_NOISE: https://arxiv.org/pdf/1509.02971.pdf

4- OAC: https://arxiv.org/pdf/1910.12807.pdf

5- FIGAR: https://arxiv.org/pdf/1702.06054.pdf

6- SAC_POLYRL / DDPG_POLYRL: https://arxiv.org/pdf/2012.13658.pdf

In the directory  ``engine/reward_modifier``, we define different sparsity levels for the MuJoCo environments.

The code to run the program is as follows:

``` python3 main.py```

If you want to change the environment you should type:

``` python3 main.py --env_name Ant-v2```
The environments by defualt are non-sparse. You can make the reward sparse by using the following command:

``` python3 main.py --env_name Ant-v2 --sparse_reward --threshold_sparsity 0.05```
where ``--sparse_reward`` makes the environment's reward sparse and ``--threshold_sparsity`` determines the 
extent of sparsity.

If you want to change the algorithm: 

``` python3 main.py --env_name --algo SAC```

Defualt algorithm is DDPG_POLYRL. Current supported algorithms are: ``--algo SAC, -algo SAC_POLYRL --algo DDPG,
 --algo DDPG_PARAM, --algo DDPG_NO_NOISE, --algo DDPG_OU_NOISE, --algo DDPG_POLYRL, --algo OAC, --algo FIGAR``

If you want to change the parameters of a specific algorithm (for example SAC) you should write:

``` python3 main.py --env_name --algo SAC --tau_sac 1```

which changes the ``tau `` parameter of SAC algorithm. Detailed information regarding the different setting of 
algorithms can be found in main.py argparser.
