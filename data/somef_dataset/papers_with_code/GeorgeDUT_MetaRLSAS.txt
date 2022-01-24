# Meta_RL_For_SAS
## TooL
Meta_RL_for_SAS:

This project is using Meta Reinforcement learning to enhance the adaptability of self-learning adaptive system (SLAS).

The process of optimizing and the adaptation process from meta parameters to optimal parameters:

![figure](MRL.png)

(This is adapted from MAML paper.)

## Getting started
To avoid any conflict with your existing Python setup, and to keep this project self-contained, it is suggested to work in a virtual environment with [`virtualenv`](http://docs.python-guide.org/en/latest/dev/virtualenvs/). To install `virtualenv`:
```
pip install --upgrade virtualenv
```
Create a virtual environment, activate it and install the requirements in [`requirements.txt`](requirements.txt).
```
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Requirements
 - Python 3.5 or above
 - PyTorch 1.3
 - Gym 0.15

### Usage

#### Training
You can use the [`train.py`](train.py) script in order to run reinforcement learning experiments with MAML. Note that by default, logs are available in [`train.py`](train.py) but **are not** saved (eg. the returns during meta-training). For example, to run the script on mdp-complex:
```
python train.py --config configs/maml/mdp/mdp-complex.yaml --output-folder mdp-complex/ --seed 2 --num-workers 4
```

#### Testing
Once you have meta-trained the policy, you can test it on the same environment using [`test.py`](test.py):
```
python test-my-new.py --config mdp-complex/config.json --policy mdp-complex/policy.th --output mdp-complex/results.npz --meta-batch-size 1 --num-batches 2 --num-workers 2
```

Grad_Steps = 50 in test-my-new.py is the step you want to print. You can set it to any number.
But you should keep Grad_Steps < num-steps (num-steps is set in mdp-complex/config.json)


We already save a trained model in mdp-complex.

#### How to change the parameters and use yourself environment
1) maml_rl/envs/__init__.py
	register your environment:
	
	register(
    'TabularMDP-v1',
    entry_point='maml_rl.envs.mdp-my:TabularMDPEnv',
    kwargs={'num_states': 5, 'num_actions': 3},
    max_episode_steps=10
)

2) basic settings:
/configs/maml/mdp/mdp-my.yaml

3) change the environment:
/maml_rl/envs/mdp-my.py

4) train & test

## Algorithm: MAML
Our basic algorithm is based on MAML:

https://github.com/tristandeleu/pytorch-maml-rl

MAML project is, for the most part, a reproduction of the original implementation [cbfinn/maml_rl](https://github.com/cbfinn/maml_rl/) in Pytorch. These experiments are based on the paper
> Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-Agnostic Meta-Learning for Fast Adaptation of Deep
Networks. _International Conference on Machine Learning (ICML)_, 2017 [[ArXiv](https://arxiv.org/abs/1703.03400)]


Reinforcement Learning with Model-Agnostic Meta-Learning (MAML)
Implementation of Model-Agnostic Meta-Learning (MAML) applied on Reinforcement Learning problems in Pytorch. This repository includes environments introduced in ([Duan et al., 2016](https://arxiv.org/abs/1611.02779), [Finn et al., 2017](https://arxiv.org/abs/1703.03400)): multi-armed bandits, tabular MDPs, continuous control with MuJoCo, and 2D navigation task.


## References

If you want to cite this implementation of MetaRLSAS:
```
@inproceedings{mingyue21ameta,
  author    = {Mingyue Zhang and Jialong Li and Haiyan Zhao and Kenji Tei and Shinichi Honiden and Zhi Jin},
  title     = {A Meta Reinforcement Learning-based Approach for Online Adaptation},
  booktitle = {{IEEE} International Conference on Autonomic Computing and Self-Organizing
               Systems, {ACSOS} 2021},
  publisher = {{IEEE}},
  year      = {2021}
}
```