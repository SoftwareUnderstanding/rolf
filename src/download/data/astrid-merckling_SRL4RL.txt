


# XSRL (eXploratory State Representation Learning)
[[Paper]](https://arxiv.org/pdf/2109.13596.pdf)
[[Project Page]](https://www.astrid-merckling.com/publication/xsrl/)
[[Watch a presentation]](https://youtu.be/bv01X2peShU?t=1300)


This is the official PyTorch implementation of the paper titled "**Exploratory State Representation Learning**". \
If you find this useful for your research, please use the following citation.
```
@misc{merckling2021exploratory,
      title={Exploratory State Representation Learning}, 
      author={Astrid Merckling and Nicolas Perrin-Gilbert and Alexandre Coninx and Stéphane Doncieux},
      year={2021},
      eprint={2109.13596},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

XSRL is a *state representation learning (SRL)* method which consists in pretraining state representation models from image observations, in a reward-free environment.
Unlike the SRL methods developed so far, none has yet developed an exploration strategy as our method does.
For more details on the contributions made with XSRL, see [Project Page](https://www.astrid-merckling.com/publication/xsrl/) or [Paper](https://arxiv.org/pdf/2109.13596.pdf).


# Getting started

## Prerequisites

- Ubuntu with anaconda
- CPU or NVIDIA GPU + CUDA CuDNN

## Installation


1. To use the same PyBullet environments as in our experiments, install the [`bullet_envs`](https://github.com/astrid-merckling/bullet_envs) package:
```bash
cd <installation_path_of_your_choice>
git clone https://github.com/astrid-merckling/bullet_envs.git
cd bullet_envs
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

2. Clone this repo and have its path added to your `PYTHONPATH` environment variable:
```bash
cd <installation_path_of_your_choice>
git clone https://github.com/astrid-merckling/SRL4RL.git
cd SRL4RL
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

3. Create and activate conda environment:
```bash
cd SRL4RL
# for CPU only
conda env create -f environment_cpu.yml
# for CPU and GPU
conda env create -f environment.yml
conda activate SRL4RL
```

The environment should be ready to run.
See the following sections for instructions of how to pretrain SRL models and train *reinforcement learning (RL)* policies.


To deactivate and remove the conda environment:
```bash
conda deactivate
conda env remove --name  SRL4RL
```

# Instructions

## XSRL Training

To train a XSRL model with a state representation of dimension 20 run the following commands.

For the regular XSRL method:
```bash
mpirun -np 1 python -u -W ignore SRL4RL/xsrl/scriptXSRL.py \
    --state_dim 20 \
    --env_name {TurtlebotMazeEnv-v0,ReacherBulletEnv-v0,HalfCheetahBulletEnv-v0,InvertedPendulumSwingupBulletEnv-v0} \
    --autoEntropyTuning True \
    --weightLPB 1 \
    --weightInverse 0.5 \
    --resetPi True
```

For XSRL-MaxEnt ablation:
```bash
mpirun -np 1 python -u -W ignore SRL4RL/xsrl/scriptXSRL.py \
    --state_dim 20 \
    --env_name {TurtlebotMazeEnv-v0,ReacherBulletEnv-v0,HalfCheetahBulletEnv-v0,InvertedPendulumSwingupBulletEnv-v0} \
    --weightEntropy 1
```

For XSRL-random ablation:
```bash
mpirun -np 1 python -u -W ignore SRL4RL/xsrl/scriptXSRL.py \
    --state_dim 20 \
    --env_name {TurtlebotMazeEnv-v0,ReacherBulletEnv-v0,HalfCheetahBulletEnv-v0,InvertedPendulumSwingupBulletEnv-v0}
```


This will produce 'logsXSRL/hashCode' folder, where all the outputs are going to be stored including train/test logs, train/test videos, and PyTorch models.


### *XSRL Representations Evaluation*

To evaluate a XSRL model during/after training run:
```bash
dir=<XSRL_trained_model_path>
mpirun -np 1 python -u -W ignore SRL4RL/xsrl/evalXSRL.py \
    --dir $dir
```

## RAE Training

To train a RAE model ([Regularized Autoencoder](https://arxiv.org/abs/1903.12436)) with an effective exploration (`--randomExplore True`) on one of the environment from image-based observations, run the following command:
```bash
mpirun -np 1 python -u -W ignore SRL4RL/ae/scriptAE.py \
    --state_dim 20 \
    --env_name {TurtlebotMazeEnv-v0,ReacherBulletEnv-v0,HalfCheetahBulletEnv-v0,InvertedPendulumSwingupBulletEnv-v0} \
    --method RAE \
    --randomExplore True
```


This will produce 'logsRAE/hashCode' folder, where all the outputs are going to be stored including train/test logs, train/test videos, and PyTorch models.

### *RAE Representations Evaluation*

To evaluate a RAE model during/after training, run:
```bash
dir=<RAE_trained_model_path>
mpirun -np 1 python -u -W ignore SRL4RL/ae/evalAE.py \
    --dir $dir
```


## Representations Transfer with Reinforcement Learning

In order to perform a quantitative evaluation of state estimators pretrained with our algorithm (XSRL) or other baselines, we validate the effectiveness of state representations as inputs to RL systems to solve unseen control tasks.
To do so, we use the deep RL algorithm SAC ([Soft Actor-Critic](https://arxiv.org/pdf/1812.05905.pdf)), which has shown promising results on the standard continuous control tasks `InvertedPendulum` and `HalfCheetah`.
The SAC implementation is based on [pytorch-soft-actor-critic](https://github.com/pranz24/pytorch-soft-actor-critic).


To train a SAC+XSRL or SAC+RAE policy on one of the environment task **with image-based observations**, run:
```bash
mpirun -np 1 python -u -W ignore SRL4RL/rl/train.py \
    --srl_path <trained_model_path>
```

To train a SAC+random policy on one of the environment task **with image-based observations**, run the following command:
```bash
mpirun -np 1 python -u -W ignore SRL4RL/rl/train.py \
    --env_name {TurtlebotMazeEnv-v0,ReacherBulletEnv-v0,HalfCheetahBulletEnv-v0,InvertedPendulumSwingupBulletEnv-v0} \
    --method random_nn
```

To train a policy with other baselines (i.e. SAC+ground truth or SAC+open-loop) on one of the environment task **without image-based observations**, run the following command:
```bash
mpirun -np 1 python -u -W ignore SRL4RL/rl/train.py \
    --env_name {TurtlebotMazeEnv-v0,ReacherBulletEnv-v0,HalfCheetahBulletEnv-v0,InvertedPendulumSwingupBulletEnv-v0} \
    --method {ground_truth,openLoop}
```


This will produce 'logsRL/hashCode' folder, where all the outputs are going to be stored including train/test logs, and PyTorch models.


### *RL Policies Evaluation*

To compute the episode returns averaged over 100 episodes (`--n_eval_traj 100`) with the best RL policy (`--model_type model_best`) during/after training, run:
```bash
dir=<RL_trained_model_path>
mpirun -np 1 python -u -W ignore SRL4RL/rl/demo.py \
    --dir $dir \
    --model_type model_best \
    --n_eval_traj 100
```


To record an evaluation video (`--save_video True`) and all image frames (`--save_image True`), with a good resolution (`--highRes True`), with the best RL policy (`--model_type model_best`) during/after training, run:
```bash
dir=<RL_trained_model_path>
mpirun -np 1 python -u -W ignore SRL4RL/rl/demo.py \
    --dir $dir \
    --model_type model_best \
    --save_video True \
    --save_image True \
    --highRes True
```

To view the policy during/after training, run:
```bash
dir=<RL_trained_model_path>
mpirun -np 1 python -u -W ignore SRL4RL/rl/demo.py \
    --dir $dir \
    --model_type model_best \
    --renders True
```


## Results

The results are presented succinctly in [Project Page](https://www.astrid-merckling.com/publication/xsrl/), for a more complete presentation and explanation of our experiments see [Paper](https://arxiv.org/pdf/2109.13596.pdf) and [XSRL chapter](https://youtu.be/bv01X2peShU?t=1300) of my PhD Defense Presentation.


# Credits

Astrid Merckling: designed the proposed algorithm, implemented the experiments, and also wrote the article.
Stéphane Doncieux, Alexandre Coninx and Nicolas Perrin-Gilbert: supervised the project and provided guidance and feedback, and also helped with the writing of the article.

```
@misc{merckling2021exploratory,
      title={Exploratory State Representation Learning}, 
      author={Astrid Merckling and Nicolas Perrin-Gilbert and Alexandre Coninx and Stéphane Doncieux},
      year={2021},
      eprint={2109.13596},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
