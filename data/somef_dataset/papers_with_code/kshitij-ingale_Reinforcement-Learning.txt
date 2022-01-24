# Reinforcement-Learning

Reinforcement learning is a class of machine learning problems which involve learning to act in an environment. The machine learning agent learns to perform actions based on the reward (feedback), it receives from the environment such that it maximizes certain criteria (usually, discounted future rewards obtained). Based on the objective and/or type of environment, there are various reinforcement learning algorithms, some of them are as follows:  

- [Dynamic Programming based algorithms](01_Dynamic_Programming/):  
These planning based methods assume complete knowledge of environment and exploit that to evaluate each state encountered in environment. Based on this evaluation, agent can learn a policy to act in the environment such that required criterion (cumulative rewards obtained with the policy) is maximized. 

- [Monte Carlo](02_Monte_Carlo/):  
These methods rely on simulating many episodes exploring different actions in an environment in order to approximate value of states encoutered during simulation. This can be then be used to learn a policy maximizing expected rewards in environment. Note that this method don't require complete knowledge of model, however, it does rely on simulating agent in environment many times which may not be always feasible.

- [Temporal Difference learning (TD-learning)](03_TD_Learning/):  
These methods are like a combination of Monte-Carlo and Dynamic programming based planning. In this method, agent interacts with environment for a few steps and later uses bootstrapping with planning based method to get expected future return at the states encountered. In this way, after evaluating different states, agent learns to choose actions which maximize the required criterion.

- [Deep Q networks (DQN)](04_Deep_Q_Networks/):  
These methods bring in the function approximators to predict expected future returns for each action in a given state of environment. The agent can then learn a policy to perform actions in the encountered states to maximize future rewards.

- [Policy gradient](05_Policy_Gradients):  
These methods attempt to directly learn the policy as opposed to earlier approaches of evaluating states and performing actions with best states. Policy gradients algorithms parameterize policy with function approximators and learn parameters such that agent learns to execute actions that maximize expected returns from states encountered in the environment.

- Trust Region Policy Optimization (TRPO):  
This is like an extension to vanilla policy gradients focusing more on step size for updates to parameters of policy appoximators ensuring better performance on updates to policy parameters.

- Proximal Policy Optimization (PPO):  
This is another extension to vanilla policy gradients addressing the step size aspect with another approach.
