# Reinforcement-Learning
Reinforcement learning homework IBIO4615

Dependencies
- Python3.5+
- Pytorch 1.0.1.
- TensorFlow 1.2
- gym, matplotlib, numpy, tensorboardx

```bash
pip install gym
pip install tensorboardx 
pip install tensorflow=1.2
```
# DQN
- Original paper: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

Tasks:
1. Play with the hyperparameters and show their corresponding graphs. Which parameter caused the most change? Which one didn’t affect that much? Discuss briefly your results
2. Anneal the 𝞮 hyperparameter to decay linearly instead of being fixed? Did it help at all? Why?
3. Try two different architectures and report any results

# DDPG
- Original paper: https://arxiv.org/abs/1509.02971
- OPENAI Baselines post: https://blog.openai.com/better-exploration-with-parameter-noise/

Tasks:
1. Change DDPG to Mountain car (May tune a bit the hyperparameters as constant time systems are different). Compare with DQN (# of episodes till convergence)
2. (Optional) As you see reward/cost penalize control law/actions change it so it penalize more control energy used and plot u(t) for different initial positions of the pendulum.

**Note that DDPG is feasible about hyper-parameters. You should fine-tuning if you change to another environment.**

Episode reward in Pendulum-v0:  

![ep_r](https://github.com/sweetice/Deep-reinforcement-learning-with-pytorch/blob/master/Char05%20DDPG/DDPG_exp.jpg)  

