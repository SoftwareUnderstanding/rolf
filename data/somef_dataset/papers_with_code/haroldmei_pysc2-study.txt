## pysc2-study

## Course project, Play StartCraft II with deep reinforcement learning  
Deep Reinforcement Learning with A3C and DQN algorithms.  

#### A3C/DQN score plots. The slow training speed of DQN makes it look like it's reaching a sub-optimal, but it's actually still improving, very slow (MoveToBeacon and FindAndDefeatZerglings).
![A3C/DQN score](https://github.com/haroldmei/pysc2-study/blob/master/experiments/scores.png)




Deep Q-Network: https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf    
Asynchronous A3C and DQN: http://arxiv.org/abs/1602.01783  
SC2LE: http://arxiv.org/abs/1708.04782  
Code adaptered from https://github.com/deepmind/pysc2, https://github.com/xhujoy/pysc2-agents and other github projects;  


Environment: Windows 10 + 8G nVidia GPU, Debian GNU/Linux 9.9 + 8G Tesla GPU   
TensorFlow 1.13
PySC2 2.02
Python 3.5+


### Experiments on:
Asynchronous Advanced Actor-Critic with Deep Convolutional Neural Network  
Asynchronous Q learning with Deep Convolutional Neural Network  

Training with A3C agent:  
python agent.py --map=MoveToBeacon --agent agents.a3c_agent.A3CAgent  

Training with DQN agent:  
python agent.py --map=MoveToBeacon --agent agents.dqn_agent.DeepQAgent  

Training with Random agent, for comparison:  
python agent.py --map=MoveToBeacon --agent agents.random_agent.RandomAgent

use --max_agent_steps to make longer trajectory if you have a powerful computer:  
python agent.py --map=FindAndDefeatZerglings --max_agent_steps=1000 

use --training=False to evaluate, will only start a single thread.:  
python agent.py --map=MoveToBeacon --training=False

--map set to other mini games as well:  
python agent.py --map=DefeatRoaches

After pysc2 installed, the training can be also run in this way  
python -m pysc2.bin.agent --map CollectMineralShards --agent agents.a3c_agent.A3CAgent  

python agent.py --map=MoveToBeacon --max_agent_steps=120 --agent=agents.dqn_agent.DeepQAgent --continuation=True

python agent.py --map=MoveToBeacon --agent=agents.dqn_agent.DeepQAgent --training=False
