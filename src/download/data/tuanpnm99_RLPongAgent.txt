# Reinformence Learning Pong Agent


This project is created to train RL Agent to learn how to play Atari Pong Game using Policy Gradient methods from only raw input game pixels. First, I initialize the policy for the RL agent to play Pong. It is more or less random. Then, I gather the data with respect to the current policy by playing the game. I process the screen input and used CNN (Convolutional Neural Network) to extract meaningful information about the game state. Using this game state information with the current policy, I get the next action. The agent plays for a while until it stops and improves its current policy using the Policy Gradient method with the collected data.

Specifically, the Policy Gradient method I used for this agent is Proximal Policy Optimization 
https://arxiv.org/abs/1707.06347

After 3 days of training, this Reinforcement Learning Pong Agent was able to beat the hardcoded Artificial Intelligence Pong Agent 
