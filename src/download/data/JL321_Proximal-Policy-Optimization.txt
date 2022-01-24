# Proximal-Policy-Optimization
Tensorflow implementation of PPO from (https://arxiv.org/abs/1707.06347). Without any changed parameters, the program trains an agent in the Humanoid-v2 environment from OpenAI Gym.

**Dependencies:**
- Mujoco-py (Mujoco150+) 
- OpenAI gym
- Numpy
- Tensorflow
- Matplotlib
- Scipy

**Usecase:**
Example call:
```
python ppoMain.py --env Humanoid-v2 --episodes 1000 --localsteps 2000 --batchSize 64
```

`python ppoMain.py -h`can be used to learn more about the input format.
