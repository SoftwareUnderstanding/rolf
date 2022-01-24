### 基于强化学习的智能定价

------

#### 参考

DQN: https://arxiv.org/pdf/1312.5602.pdf

DDQN: https://arxiv.org/pdf/1509.06461.pdf

背景：https://github.com/1156564544/smart-pricing-based-on-reinforcement-learning

------

#### 设置

**state: [remaining_time, utilization, price]**

remaining_time是剩余天数/(截止日期-开售日期)

utilization是舱位利用率，即1-剩余舱位/总舱位

price设为(pt-p0)/p0，pt是当前价格，p0是最低售价，price取值[0,1] (需要根据真实情况再调整)

**action: [-5%, -4%, ..., 4%, 5%]**

共11个动作。-5%为在现定价基础上减少5%。（训练时动作空间可通过改变参数调整。）

**reward: price*售出舱位数**

------

#### 结构

采用了Double Deep Q Networks模型，详见参考论文&代码注释。

`dqn_s.py` : 定义了DQNAgent

DQNAgent可以根据 **state** 来判断 **action**。它用隐藏层大小为[16, 16]的神经网络来训练和预测。

`env_1.py` : 定义了环境Env

在每一段时间内，顾客会根据DQNAgent给出的定价决定是否购买舱位（顾客数量服从泊松分布）

在DQNAgent做出action后，Env会根据顾客购买情况返回相应的next state，reward和done。done取值0或1，反映本航线是否结束销售；当剩余时间为0或存货为0时，done=1。

`run_dqn.py` : 设置Env，对DQNAgent进行训练

------

#### 后续

`pricing.py` : 搬运https://github.com/1156564544/smart-pricing-based-on-reinforcement-learning 的`comparing.py`。用训练出来的DQNAgent进行定价，并和随机/固定定价对比。

现训练得的DQNAgent在某些设定中表现不如固定定价（可能训练时间还不够），且它更倾向于降价... 这个可能跟Env设置有关......
