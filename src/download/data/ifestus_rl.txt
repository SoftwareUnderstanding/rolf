# My Reinforcement Learning Implementations

## Currently focused on:
DQN (and DDQN)
REINFORCE with baseline
PPO
TRPO


`I wanted to start with the easier/basic methods (as described in Reinforcement Learning: an Introduction). One issue I've run into, although I'm probably missing something, is that (at least for value-functions) full representations of the state space are required. It's not 100% clear to me how to represent that simply while taking advantage of the provided environments in openai/gym. My plan for now is to start with the above three methods, and then I'll write some specific environments and representations for the other methods.`

## Tabular Methods:
1. Rollout
2. Monte-Carlo Tree Search

## Control Goals
1. n-step SARSA on-policy (probably start with 1-step)
2. n-step SARSA off-policy
3. n-step Tree Backup
5. Q-learning
5. n-step Q(sigma)

## TD(lambda)

## On-Policy Estimation with Approximation
1. Gradient Monte-Carlo
2. Semi-Gradient TD(0)

## On-Policy Control with Approximation
1. n-step semi-gradient SARSA
2. n-step differential semi-gradient SARSA

## Off-Policy Control with Approximation
0. DQN (paper: https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) -- DDQN (paper: https://arxiv.org/pdf/1509.06461.pdf)
1. GTD(0)
2. Semi-Gradient TD(lambda)

## Policy Optimization Goals:
1. REINFORCE
2. REINFOCE with Baseline
3. PPO (paper: https://arxiv.org/abs/1707.06347)
4. TRPO (paper: https://arxiv.org/abs/1502.05477)

## Extra
1. GoZero (paper: https://www.nature.com/articles/nature24270)

### Unless explicitly stated, these algorithms are my implementations of the material in `Reinforcement Learning: an Introduction` by Sutton and Barto