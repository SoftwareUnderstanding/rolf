# DDPG

Continuous control with deep reinforcement learning

[http://arxiv.org/abs/1509.02971](http://arxiv.org/abs/1509.02971)

The goal of these algorithms is to perform policy iteration
by alternatively performing policy evaluation 
on the current policy with Q-learning, and then improving upon the
current policy by following the policy gradient.
<table>
  <tr>
    <td><img src="https://github.com/yusme/DDPG/blob/master/3D33.png" width="300"></td>     
    <td><img src="https://github.com/yusme/DDPG/blob/master/3D35.png" width="300"></td>
    <td><img src="https://github.com/yusme/DDPG/blob/master/episode2D.png" width="300"></td>
  </tr>
</table>


## TODO

- Batch Normalization 
- Prioritized Experience Replay (https://arxiv.org/abs/1511.05952): to replay important transitions from reply Memory more frequently


## Reference 

- [Continuous control with Deep Reinforcement Learning (by TP Lillicrap)](http://arxiv.org/abs/1509.02971)
- [Deterministic Policy Gradients (by TP D. Silver )](http://proceedings.mlr.press/v32/silver14.pdf)
- [Human-level control through deep reinforcement learning (by V. Mnih )](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf)







