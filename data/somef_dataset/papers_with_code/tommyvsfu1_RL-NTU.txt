# RL-NTU

## PG - PPO  
### PPO paper
https://arxiv.org/pdf/1707.06347.pdf
>我現在的理解是，PPO 會用到AC, A2C的技術，也就是advantage function不只是discount rewards，會使用Q,V來實作(AC技術），也會將V, pi併在一起變成一個大的network，這就是為何PPO實作上，會很像A2C的寫法。總結一句話，個人認為PPO在實作上就是A2C + surrogate loss + important sampling 技術


## DQN - BreakoutNoFrameskip-v4  
### Preprocessing
Use deepmind wrapper to do some preprocessing of game images, check this awesome blog [Frame Skipping and Pre-Processing for Deep Q-Networks on Atari 2600 Games
](https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/).

### tips 

1. MAX_STEP 跟 Replay Buffer size 
如果MAX_STEP 太大比如While(True), 那Replay Buffer 很快就會滿，且無法收集到過去的資料集


## references

papers  
[Policy Gradient 2000](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf)

https://github.com/AdrianHsu/breakout-Deep-Q-Network
https://github.com/JasonYao81000/MLDS2018SPRING
https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO.py


Pong
https://wpovell.net/posts/pg-pong.html
