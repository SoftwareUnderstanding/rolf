# Assessing the Influence of Models on the Performance of Reinforcement Learning Algorithms applied on Continuous Control Tasks

This is the master thesis project by Giacomo Arcieri, written at the [FZI](https://www.fzi.de/startseite/) Research Center for Information Technology (Karlsruhe, Germany). 

## Introduction

Model-Based Reinforcement Learning (MBRL) has recently become popular as it is expected to solve RL problems with fewer trials (i.e. higher sample efficiency) than model-free methods. However, it is not clear how much of the recent MBRL progress is due to improved algorithms or due to improved models. Hence, this work compares a set of mathematical methods that are commonly used as models for MBRL. This thesis aims to provide a benchmark to assess the model influence on RL algorithms. The evaluated models will be (deterministic) Neural Networks (NNs), ensembles of (deterministic) NNs, Bayesian Neural Networks (BNNs), and Gaussian Processes (GPs). Two different and innovative BNNs are applied: the [Concrete Dropout](https://arxiv.org/abs/1705.07832) NN and the [Anchored Ensembling](https://arxiv.org/abs/1810.05546). The model performance is assessed on a large suite of different benchmarking environments, namely one [OpenAI Gym](https://github.com/openai/gym) Classic Control problem (Pendulum) and seven [PyBullet-Gym](https://github.com/benelot/pybullet-gym) tasks (MuJoCo implementation). The RL algorithm the model performance is assessed on is Model Predictive Control (MPC) combined with Random Shooting (RS).

## Requirements

This project is tested on Python 3.6.

First, you can perform a minimal installation of OpenAI Gym with

```bash
git clone https://github.com/openai/gym.git
cd gym
pip install -e .
```

Then, you can install Pybullet-Gym with 

```bash
git clone https://github.com/benelot/pybullet-gym.git
cd pybullet-gym
pip install -e .
```

*Important*: Do not use ```python setup.py install``` or other Pybullet-Gym installation methods.

Finally, you can install all the dependencies with 

```bash
pip install -r requirements.txt
```

**Important**: There are a couple of changes to make in two Pybullet-Gym envs:
1) There is currently a mistake in Hopper. This project uses HopperMuJoCoEnv-v0, but this env imports the Roboschool locomotor instead of the MuJoCo locomotor. Open the file
```
pybullet-gym/pybulletgym/envs/mujoco/envs/locomotion/hopper_env.py
``` 
and change 
```
from pybulletgym.envs.roboschool.robots.locomotors import Hopper
``` 
with 
```
from pybulletgym.envs.mujoco.robots.locomotors.hopper import Hopper
```

2) Ant has ```obs_dim=111``` but only the first 27 obs are important, the others are only zeros. If it is true that these zeros do not affect performance, it is also true they slow down the training, especially for the Gaussian Process. Therefore, it is better to delete these unimportant obs. Open the file
```
pybullet-gym/pybulletgym/envs/mujoco/robots/locomotors/ant.py
``` 
and set ```obs_dim=27``` and comment or delete line 25
```
np.clip(cfrc_ext, -1, 1).flat
```

## Project Description

### Models

The models are defined in the folder ```models```:

- ```deterministicNN.py```: it includes  the deterministic NN (```NN```) and the deterministic ensemble (```ens_NNs```).

- ```PNN.py```: here the Anchored Ensembling is defined following this [example](https://github.com/TeaPearce/Bayesian_NN_Ensembles). ```PNN``` defines one NN of the Anchored Ensembling. This is needed to define ```ens_PNNs``` which is the Anchored Ensembling as well as the model applied  in the evaluation.

- ```ConcreteDropout.py```: it defines the Concrete Dropout NN, mainly based on the Yarin Gal's [notebook](https://github.com/yaringal/ConcreteDropout/blob/master/concrete-dropout-keras.ipynb), but also on this other [project](https://github.com/exoml/plan-net). First, the ConcreteDropout Layer is defined. Then, the Concrete Dropout NN is designed (```BNN```). Finally, also an ensemble of Concrete Dropout NNs is defined (```ens_BNN```), but I did not use it in the model comparison (```ens_BNN``` is extremely slow and ```BNN``` is already like an ensemble).

- ```GP.py```: it defines the Gaussian Process model based on [gpflow](https://github.com/GPflow/GPflow). Two different versions are applied: the ```GPR``` and the ```SVGP``` (choose by setting the parameter ```gp_model```). Only the ```GPR``` performance is reported in the evaluation because the ```SVGP``` has not even solved the Pendulum environment.

### RL algorithm

The model performance is evaluated in the following files:

1) ```main.py```: it is defined the function ```main``` which takes all the params that are passed to ```MB_trainer```. Five ```MB_trainer``` are initialized, each with a different seed, which are run in parallel. It is also possible to run two models in parallel by setting the param ```model2``` as well. 

2) ```MB_trainer.py```: it includes the initialization of the env and the model as well as the RL training loop. The function ```play_one_step``` computes one step of the loop. The model is trained with the function ```training_step```. At the end of the loop, a pickle file is saved, wich includes all the rewards achieved by the model in all the episodes of the env.

3) ```play_one_step.py```: it includes all the functions to compute one step (i.e. to choose one action): the epsilon greedy policy for the exploration, the Information Gain exploration, and the exploitation of the model with MPC+RS (function ```get_action```). The rewards as well as the RS trajectories are computed with the cost functions in ```cost_functions.py```.

4) ```training_step.py```: first the relevant information is prepared by the function ```data_training```, then the model is trained with the function ```training_step```.

5) ```cost_functions.py```: it includes all the cost functions of the envs. 

Other two files are contained in the folder ```rewards```:

- ```plot_rewards.ipynb```: it is the notebook where the model performance is plotted. First, the 5 pickles associated with the 5 seeds are combined in only one pickle. Then, the performance is evaluated with various plots. 

- ```distribution.ipynb```: this notebook inspects the distribution of the seeds in InvertedDoublePendulum (Section 6.9 of the thesis).


## Results

Our results show significant differences among models performance do exist. 

It is the Concrete Dropout NN the clear winner of the model comparison. It reported higher sample efficiency, overall performance and robustness across different seeds in Pendulum, InvertedPendulum, InvertedDoublePendulum, ReacherPyBullet, HalfCheetah, and Hopper. In Walker2D and Ant it was no worse than the others either. 

Authors should be aware of the differences found and distinguish between improvements due to better algorithms or due to better models when they present novel methods. 

The figures of the evaluation are reported in the folder ```rewards/images```.

## Acknowledgment

Special thanks go to the supervisor of this project David Woelfle. 
