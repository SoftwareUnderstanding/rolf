# CS-221-Final-Project

The primary goal of this research  is to find the best __Fee__ mechanism for trades on the blockchain. Reinforcement Learning (RL) agents acts as people under certain Fee policies. We observe how RL agents's behavior changes with Fee mechanism changes and observe the total fee and volume of trades made by the agent in the trading environment.

# Structure
1.  model.py and PPO.ipynb
    store trading agents and specify how to train the agents and how to use them. 
2. [data](https://github.com/amzoyang/CS-221-Final-Project/blob/master/SampleDataset.ipynb)
    Stores the historical data to train the agents
3.  DerivativeUtils.py and TradingEnvIntegrated.py
    store the environment where fee different fee mechanisms applied

# Simulation Method
 1. **Train RL agents** using [trading gym](https://github.com/Yvictor/TradingGym/). 

 2. **Transfer agents** to different environments where different fee mechanism is applied. 
 Agents will trained again for 500 episodes more to adapt to each environment. Also, differentiate agents by varying risk_aversion ratio so that some agents prefer risk while others not.

 3. **Observe** how agents behave in each environment. Especially watch the total_volume and total_fee from each environment. Derive insights from the observation what characteristics of fee mechanism makes the difference.

# Adopted Fee mechanisms 
1. No fee
2. fee = 0.003 (0.3%)
3. fee = 0.005 (0.5%)
4. Bollinger band bound Environment
5. RSI bound Environment
6. MACD bound Environment
7. Stochastic slow bound Environment


# Used Algorithms for trading agents
## PPO
https://arxiv.org/abs/1707.06347
## Rainbow
https://arxiv.org/abs/1710.02298
## Attention
http://nlp.seas.harvard.edu/2018/04/03/attention.html

