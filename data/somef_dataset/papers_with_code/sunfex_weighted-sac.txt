## Description
------------
A weighted version of the [Soft Actor-Critic Algorithms](https://arxiv.org/pdf/1801.01290.pdf). The code is largely derived from a [Pytorch SAC Implementation](https://github.com/pranz24/pytorch-soft-actor-critic)

We support five sampling strategies:

- Uniform Sampling (as used in the original version of SAC)
- [Emphasizing Recent Experience (ERE)](https://arxiv.org/abs/1906.04009)
- Approximated version of ERE (ERE\_apx, Proposition 1 in our paper)
- 1/age weighed sampling
- [Prioritized Experience Replay (PER)](https://arxiv.org/abs/1511.05952) 

This implementation demonstrates that ERE, ERE\_apx and 1/age share very similar performances and are better than uniform sampling and PER.

## Requirements
------------

- [mujoco-py](https://github.com/openai/mujoco-py)
- [TensorboardX](https://github.com/lanpa/tensorboardX)
- [PyTorch](http://pytorch.org/)


## Usage

#### For uniform sampling, PER, ERE\_apx, and 1/age:

Uniform Sampling:
```
python main.py --mode SAC
```

Prioritized Experience Replay (PER):
```
python main.py --mode PER
```

ERE\_apx:
```
python main.py --mode ERE2
```

1/age Weighting:
```
python main.py --mode HAR
```

#### For ERE:

Original ERE: update K times after sampling a length-K trajectory.
```
python main.py --mode EREo
```

ERE with estimated trajectory length (update once whenever getting a new (s,a,r,s'))
```
python main.py --mode EREe
```

(Note: We use `--mode EREo` to evaluate the ERE strategy in our paper. Still, `--mode EREe` has a very similar performance)

### Arguments

```
  --env_name       Mujoco Gym environment (default: HalfCheetah-v2)
  --gamma          discount factor (default: 0.99)
  --tau            target smoothing coefficient (default: 5e-3)
  --lr             learning rate (default: 3e-4)
  --alpha          temperature parameter determines the relative importance 
                   of the entropy term against the reward (default: 0.2)
  --seed           random seed
  --batch_size     batch size (default: 512)
  --num_steps      maximum number of steps (default: 1e6)
  --hidden_size    hidden size (default: 256)
  --start_steps    steps sampling random actions (default: 1e4)
  --replay_size    size of replay buffer (default: 1e6)
  --mode           Weighting strategies (default: SAC)
```

------------

| Environment **(`--env-name`)**| Temperature **(`--alpha`)**|
| --------------- | ------------- |
| HalfCheetah-v2  | 0.2 |
| Hopper-v2       | 0.2 |
| Walker2d-v2     | 0.2 |
| Ant-v2          | 0.2 |
| Humanoid-v2     | 0.05 |
