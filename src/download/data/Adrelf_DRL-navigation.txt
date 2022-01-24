# DRL-navigation
Train an agent to navigate in a complex environment and collect bananas. The deep reinforcement learning algorithm is based on value-based method (DQN).

![alt text](https://github.com/Adrelf/DRL-navigation/blob/master/images/banana.gif)

# The Environment 
The environment is determinist.
 + State: 
 The state space has 37 dimensions and contains:
    - 7 rays projecting from the agent at the following angles: [20, 90, 160, 45, 135, 70, 110] # 90 is directly in front of the agent
    - Each ray is projected into the scene. If it encounters one of four detectable objects the value at that position in the array is set to 1. Finally there is a distance measure which is a fraction of the ray length: [Banana, Wall, BadBanana, Agent, Distance]
    - the agent's velocity: Left/right velocity (usually near 0) and Forward/backward velocity (0-11.2)
 + Actions:
 The action space has 4 discrete action and contains:
    - 0: move forward
    - 1: move backward
    - 2: turn left
    - 3: turn righ
 + Reward strategy:
    - +1 for collecting a yellow banana
    - -1 for collecting a blue banana
 + Solved Requirements:
Considered solved when the average reward is greater than or equal to +13 over 100 consecutive trials.

# Algorithm
DQN with some improvements:
 - Deep Reinforcement Learning with Double Q-learning ==> https://arxiv.org/abs/1509.06461 
 - Dueling Network Architecture ==> https://arxiv.org/abs/1511.06581
 - Prioritized Experience Replay ==> https://arxiv.org/abs/1511.05952
 
# Getting started
Step 1: Install ML-agents ==> https://github.com/Unity-Technologies/ml-agents and follow the instructions here ==> https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md.

Step 2: Install Python (only version >3 is supported) and PyTorch.

Step 3: Clone this repository.

Step 4: Download the Unity Environment ==> https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip
Then, place the file in the DRL-navigation/ folder in this repository, and unzip (or decompress) the file.

To train an agent, please use the following command:
$python main_navigation.py

To assess the performance of a given agent:
$python eval_navigation.py
