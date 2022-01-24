# Introduction
This is a Project for Continuous Control Deep Reinforcement Learning Nanodegree @ Udacity. The Task is to follow a target with a multijoint robot arm. A DDPG model is applied to accomplish the task. The model achieves the desired +30 score on average per episode.

# The environment
The environment is describen in detail at https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control repository. For this project the single arm task is used, where a single 2-join robot arm has to learn to follow a sphere target region. That moves around the robot arm with varying speeds, but stays within reach for the entire episode. The environment is called Reacher. The Environment is a UnityAgent environment. See video for untrained model: https://www.youtube.com/watch?v=bhsVB0QKvoQ (Vincent Tam)

# Observation and action space
The observation space has 33 variables representing position, rotation, velocity, and angular velocities of the arm. The action vector contains four continuous values, representing the torque applicable to the two joints of the arm. These values always fall between [-1, 1]. There is no mentioning in the original repository how much steps or other constraints lead to episode termination, but through testing it seems that 1000 steps is one episode. The reward for each step spent in the goal location is rewarded with +0.1. The environment is considered solved if the last 100 episodes' mean score is over +30. 

# Applied model
This is a continuous input continuous output learning problem and this solution applies DDPG (Deep Deterministic Policy Gradient) a specialized Actor Critic method developed for Continuous Control problems. DDPG applies an Actor and a Critic. The actor continuously optimalized through maximalizing the critic value for a specific action-set given in a specific state by the actor, whereas the critic is optimalized through the actor's target network and its own target network, to maintain relative independence of the optimalizing process. The target networks are updated through soft updating at each step. As a result the actor network in itself achieves a policy that approximates the optimal policy for the given task. For further implementation details see the Report.md file in this Repository. For further references see sources below.  

- Spinup documentation: https://spinningup.openai.com/en/latest/algorithms/ddpg.html
- Continuous control with deep reinforcement learning (article): https://arxiv.org/abs/1509.02971

# Results
The DDPG model implemented here converged to the target value of 30. at episode 508, with a value of 30.0793.
![Continuous Control Convergence Graph](https://github.com/petsol/ContinuousControl_UnityAgent_DDPG_Udacity/blob/master/ContinuousControl_convergence.png?raw=true)

# Rerunning the model

- Download the appropriate Unity environment for your system from above github repository.
- Create a python 3.6* environment, containing the following packages 
  - pytorch 0.4*
  - unityagents 0.4
- Clone repo
- Update environment path in you copy (Continuous_Control.ipynb)
- Run the Continuous_Control.ipynb notebook

\*(higher might work, but not guaranteed)

# Saved model parameter-sets
- al1.state (actor-local)
- at1.state (actor-target)
- cl1.state (critic-local)
- cl2.state (critic-target)
