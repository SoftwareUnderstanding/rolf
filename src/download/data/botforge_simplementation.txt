# simplementation
Clean implementations of papers I read for research in robotics + handwritten notes on some of them. It aims to offer well commented code that flows well. I vaguely structure these according to: https://spinningup.openai.com/en/latest/spinningup/keypapers.html

### 1. Human-level control through deep reinforcement learning, Mnih et al, <em>Nature</em> 2015.
* **Gist** : Use DeepRL and Deep Q-Learning (DQN) to achieve above human level performance in ATARI Games
* **Paper** : https://www.nature.com/articles/nature14236.pdf
* **Algorithm/Techniques** : DQN, Experience Replay

### 2. Deep Reinforcement Learning with Double Q-learning, van Hasselt et al, 2015.
* **Gist** : Illustrate vanilla DQN's tendency to overestimate Q-value's, and propose 'Double DQN' to use two seperate Neural Networks to select the action and evaluate its Q-value respectively. 
* **Paper** : https://arxiv.org/pdf/1509.06461.pdf
* **Algorithm/Techniques** : Double-DQN, Experience Replay

### 3. Dueling Network Architectures for Deep Reinforcement Learning, Wang et al, 2016.
* **Gist** : Propose the "Dueling Network Architecture" that computes an estimate of the value function & and an estimate of the advantage seperately to evaluate the Q-value. 
* **Paper** : https://arxiv.org/pdf/1511.06581.pdf
* **Algorithm/Techniques** : Dueling Q-Network, Experience Replay

### 4. Prioritized Experience Replay, Schaul et al, <em>ICLR</em> 2016.
* **Gist** :  
* **Paper** : ### 4. Prioritized Experience Replay, Wang et al, 2016.
* **Algorithm/Techniques** : Prioritized Experience Replay, Q-Learning 
