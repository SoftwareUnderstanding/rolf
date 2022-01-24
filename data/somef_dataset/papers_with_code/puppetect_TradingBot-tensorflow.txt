
# TradingBot-tensorflow Demo

## 目的


给定某只股票若干年的15分钟级数据，通过强化学习训练机器得到最佳操盘模型，最后用同只股票其他年份的15分钟级数据进行回测，考察其表现


## 原理
1. 数据处理（data.py），只将该股过去九天的收盘价的变化率百分比作为观察值
2. 决策模型（models.py），经过处理的数据作为观察值state导入深度学习模型得出policy
3. 环境互动（environ.py），将policy通过Agent（如epsilon-greedy或Probability selection）获得对应的动作action，并和环境互动后得到下一分钟的观察值next_state、盈亏比例reward、回合完成的指令done、和其他信息info
4. 训练模型（train.py），得到若干(next_state, reward, done, info)后，根据所选的强化学习类型（DQN或Actor-critic）计算loss并回溯优化模型参数，保存最佳参数并通过tensorboard监测模型表现



## 测试
1. 数据：平安银行（000001.SZ）2015-2018年的15分钟级后复权数据（来源：聚宽）。以前80%的数据作Train，后20%数据作Test。
2. 规则：为更好训练模型，假定为融资融券账户，可以做多做空。同时简化交易模型，规定出现多空信号后每次开仓10000股，持股3bars后自动平仓。机器动作空间为Discrete(3)，0代表不操作，1代表开空，2代表开多。
3. 框架：Tensorflow (CUDA version)
4. 平台：Google Colab (Tesla K80 GPU)
5. 策略：Feed Forward Policy
6. 模型：Double Q-Learning (https://arxiv.org/abs/1509.06461)
```
TradingDQN(
  (policy): Sequential(
    (0): Linear(in_features=9, out_features=128, bias=True)
    (1): ReLU()
    (2): Linear(in_features=128, out_features=64, bias=True)
    (3): ReLU()
    (4): Linear(in_features=64, out_features=32, bias=True)
    (5): ReLU()
    (6): Linear(in_features=32, out_features=3, bias=True)
  )
)
```

## 表现

- loss随episode缓慢降低，reward逐渐升高并最后converge，表明机器学习到了一定的规律
<img src="images/000001_loss_reward.png" >


- Train：开仓时间（蓝色开空、红色开多）与收益曲线

```
Total Trades Taken:  6969
Total Reward:  351105.5
Average Reward per Trade:  50.38
```

<img src="images/000001_history_train.png" >


- Test：开仓时间（蓝色开空、红色开多）与收益曲线

```
Total Trades Taken:  1273
Total Reward:  64043.5
Average Reward per Trade:  50.30
```

<img src="images/000001_history_test.png" >


## 优化
1. 调试训练参数，目前训练结果表明模型存在开空远多于开多的问题，需要调试参数解决
1. 增加训练数据，比如用多年或多只股票的数据反应更宏观的行情
2. 优化训练因子，提供给模型更多指标，如成交量、技术指标、基本面情况、大盘数据等
3. 尝试其他模型，如A2C+LSTM等


## 参考
1. AdrianP-/[gym_trading](https://github.com/AdrianP-/gym_trading)
2. hill-a/[stable-baselines](https://github.com/hill-a/stable-baselines)
