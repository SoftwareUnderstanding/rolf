## PPO
Proximal Policy Optimization implementation with Tensorflow.

https://arxiv.org/pdf/1707.06347.pdf

## requirements
- Python3

## dependencies
- tensorflow
- gym[atari]
- opencv-python
- git+https://github.com/imai-laboratory/lightsaber

## usage
### training
```
$ python train.py --gpu {0 or -1} --render --final-steps 10000000
```

### playing
```
$ python play.py --gpu {0 or -1} --render --load {path of models}
```

### implementation
The code base is build up on these projects

- [DQN](https://github.com/imai-laboratory/dqn)
- [OpenAI Baselines](https://github.com/openai/baselines)

### Ongoing Work
Augmenting the observation space to be aware of damage classes of the agent.
