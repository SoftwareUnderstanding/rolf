# AI-Gaming
# Problem Statement
Two RL agents, player A and player b (capital A denoting the player has the ball ) compete in an 8*8 grid environment, having 5 discrete actions ( up, down, left, right, and stationary ) to score a goal at the coordinate defined in the environment.

The agents not only have to deal with maximizing their reward in the environment but also have to deal with another RL Agent (multi-agent problem in RL).

**State-space** :
We would like to add the following variables in the state space: Ball Owner, x coordinate of player, y coordinate of player, x coordinate of the opponent, y coordinate of the opponent, (distance of the player from goal post in the x-axis ), (distance of the player from the goal in the y axis ), score of our bot, score of opponent .

**Reason** :
We want our agent to know who is the ball owner, and the distances from the goal post throughout the entire training process so that it can move to the goal post faster. Also the problem of boundaries, that is ( if our agent reaches an edge and the action sampled from our neural network forces it to move out which is nullified by the environment, the action is wasted in this case. This problem is also addressed by the last two variables in the state space that calculates the distance of the agent from the goalposts ). x and y coordinates are given as usual for the training process for offense and defense maneuvers​.

**Action space**:
We are using the actions which are possible in the environment as the action space for the problem, namely 0 - Stationary, 1 - move up, 2 - move right, 3 - move down, 4 - move left

**Rewards** :
We are treating them as values that need to be tuned for training our bot. We will just explain the rewarding system in detail here.

**Reward** - 
For Goal - a numpy array of shape (1,2) , where the first column represents the reward for our bot and the second column represents reward for the opponent (since we are using self play) .

**Penalize** - 
For losing the ball to the opponent, if our opponent is the ball owner, our bot has done self-goal, our opponent scoring goal.
Reasons for above penalizing and rewards :

**Reward** - 
We expect our agent to maximize its expected return from the environment which is done by scoring more goals. We would also like to reward the agent for having the ball as a means of telling the agent to not give the ball to the opponent

**Penalize** - 
Passing the ball to the opponent is a move that our agent cannot afford to do, and self-goal is discouraged too so that our agent is not encouraged to move backward and treat that as a goal
# Model

### The Proximal Policy Optimisation (PPO) engine:
PPO, developed by OpenAI in 2017, is a state-of-the-art RL algorithm. It updates the current version of the ​network​ being trained and fires a ​callback​ that saves the network to the bank if the current version has outperformed previous versions.

### Method
Two PPO models each for agent_A and agent_B are trained independently with a joint state space, independent action space and rewards. The idea is adopted from this [paper](https://proceedings.neurips.cc//paper/2020/file/3b2acfe2e38102074656ed938abf4ac3-Paper.pdf). 

 
# References
#### Training

https://youtu.be/gqX8J38tESw?t=2999

#### PPO
https://arxiv.org/abs/1707.06347 

https://openai.com/blog/openai-baselines-ppo/ 

https://www.youtube.com/watch?t=14m1s&v=gqX8J38tESw&feature=youtu.be
      
