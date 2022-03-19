# Introduction
In this two agent Unity environment the agents must learn to keep the ball in play and to hit the ball over the net enough times before it falls to the ground or outside the court.  An MADDPG agent is used to train the cooperating actor network policies.

# The environment
The environment is described in detail at https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet  repository. For this project two tennis rackets are to be controlled on both sides of the net. At the beginning of the episode a ball falls in (vertically) onto one of the player fields. From that point the rackets need to position themselves and/or hit the ball by "jumping" in order to pass the ball to the other agent's field. The episode ends when the ball either leaves the court or hits the court-ground. For such events a score of -0.01 is rewarded to the "faulty" agent, whereas if the ball is passed over the net an award of +0.1 is granted to the agent that passed the ball. There is no known limit to the episode steps if the ball does not fall. An episode score is considered to be the higher score of the two agents' scores. The task is solved if the last 100 scores' mean is more than +0.5. The environment is called Tennis that is a UnityAgent environment. See animation from Udacity repo:

![Multiagent Tennis Gif](https://github.com/petsol/MultiAgentCooperation_UnityAgent_MADDPG_Udacity/blob/master/tennis.gif?raw=true)


# Observation and action space
The observation space has nominally 8 variables representing position, velocity for both ball and own racket. In practice however three (probably) consecutive observations age given by the environment yielding a state size of 24 (per agent). (The ordering is not known). The action-set size is two, one action is moving towards vs. backing away from the net, and jumping. The actions are continuous. The intervals are not known, but assumed to be between [-1, 1].  

# Applied model
This is a continuous input continuous output multiagent cooperative/collaboration environment considering the reward-termination-scoring structure. For such environments MADDPG algorithm can be considered as state of the art model today. MultiAgent Deep Deterministic Policy Gradients is a multiagent version of DDPG, where not only multiple agents try to receive as much reward as possible, but their reward structure is cross propagated in a manner that reflects the needs to solve the environment. As DDPG, MADDPG applies an Actor and a Critic for each agent (in most implementations). Both actors receive only their respective state values and produce an action set on their output as a result of the learned policy. At inference this ensures that the agents can collaborate independently without direct "state-sharing". However the critics for all the agents receive all states and all actions. Since they will not play a role in inference phase the actors after training can be considered independent. There are different implementations of the MADDPG algorithm [1] In the original MADDPG paper [2] all Actors have their respective Critic network, with the other agents' (and possibly other hidden elements') state observations and actions. Both Actors are continuously optimized through maximizing the averaged Critic value for a specific action-set (for both agents), whereas the Critics are optimized through the actors' target networks (actions) and their own target networks, to maintain relative independence of the optimizing process. The target networks are updated through soft updating at each step. In this implementation the Critic networks' rewarding structures were differentiated by a "team spirit" variable. This variable adjusted the Critic networks' response to the "other" agent's rewards.  At the same time the loss function of the Actors were also sensitivized to the other agent's Critic network less if the team spirit variable was low. This deviates slightly from the original MADDPG implementation. For further implementation details see the Report.md file in this Repository. For further references see sources below.  

1. Parameter Sharing Deep Deterministic Policy Gradient for Cooperative Multiagent Reinforcement Learning: https://arxiv.org/ftp/arxiv/papers/1710/1710.00336.pdf
1. Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments: https://arxiv.org/pdf/1706.02275.pdf

# Results
The MADDPG model implemented here passed the target value of 0.5. at episode 794, with a value of 0.5079.

![Tennis Convergence Graph](https://github.com/petsol/MultiAgentCooperation_UnityAgent_MADDPG_Udacity/blob/master/Tennis_scores.png?raw=true)

# Rerunning the model

- Download the appropriate Unity environment for your system from above github repository.
- Create a python 3.6* environment, containing the following packages 
  - pytorch 0.4*
  - unityagents 0.4
- Clone repo
- Update environment path in you copy (Tennis.ipynb)
- Run the Tennis.ipynb notebook (multiagent_resources.py should be in the same (working) directory)

\*(higher might work, but not guaranteed)

# Saved model parameter-sets
- Actor1_ddpg
- Actor2_ddpg
- Actor1Target_ddpg
- Actor2Target_ddpg
- Critic1_ddpg
- Critic2_ddpg
- Critic1Target_ddpg
- Critic2Target_ddpg
