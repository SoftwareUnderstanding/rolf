# Pong
## Environment
Application of different Reinforcement Learning algorithms on the Atari game Pong.

* A2C is a synchronous variant of the A3C algorithm (https://arxiv.org/pdf/1708.05144)

* PPO (https://arxiv.org/abs/1707.06347)


## Results
### Reinforce 
#### Fully Connected
![](images/summary_reinforce_fc.png)

#### LSTM
![](images/summary_reinforce_lstm_5.png)

### A2C
![](images/summary_a2c_2.png)

### PPO
![](images/summary_ppo_fc_2.png)


## Run
###  Reinforce Fully Connected
```shell
$ python -m run_reinforce_fc
```

###  Reinforce LSTM
```shell
$ python -m run_reinforce_lstm
```

###  A2C
```shell
$ python -m run_a2c
```

###  PPO
```shell
$ python -m run_ppo_fc
```






