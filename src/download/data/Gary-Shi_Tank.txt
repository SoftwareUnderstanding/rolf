# Tank
AI Program for Tank Using Separately Deep-Q-Learning and RL+MCTS (Ongoing)



Tank_Solution3.1: 

Using Deep-Q-Learning described in https://arxiv.org/abs/1312.5602



Update:

√ Keep the enemy still and encourage our tank to attack it. Prevent our tank from learning to stay still and wait for enemy to make mistake

√ Make the enemy move without smashing into walls or slipping outside the area to fix the distribution of battles

√ Give tank a reward when marching forward to make rewards more dense



Tank_Solution4:

Using RL and MCTS described in https://www.nature.com/articles/nature24270?sf123103138=1



Update:

√ Some little adjustment to the network architecture

√ Add a resignation threshold

√ Add a Dirichlet noise in the prior probability distribution to encourage it to explore

√ Add a training dataset of winning games to fix the distribution of battles
