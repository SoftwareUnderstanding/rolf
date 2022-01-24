# 3-Layer Neural Network
Feed Forward Neural Network (FFNN)  

<img src="https://miro.medium.com/max/2636/1*3fA77_mLNiJTSgZFhYnU0Q.png" width=70%>

[cited from https://towardsdatascience.com/build-up-a-neural-network-with-python-7faea4561b31]

## Implementation
- layer-approach implementation for expanding to Deep Learning

### Layer Structure
1. Affine
2. ReLU
3. Affine
4. Softmax with Loss


### Options
#### Initialization
- Xavier (for sigmoid/tanh)
- He (for ReLU)

#### Gradient
- numerical differantial
- back propagation (based on chain-rule)

#### Learning method
- SGD
- Momentum
- AdaGrad
- Adam

#### Loss function
- sum squared error
- cross entropy error

#### Layers
- Relu layer
- Sigmoid layer
- Affine layer
- Softmax with Loss Layer


## Reference
1. 斎藤康毅, “ゼロから作るDeep Learning ーPythonで学ぶディープラーニングの理論と実装”, 初版第10刷, オーム社（オライリージャパン）, 2016.
2. 岡谷貴之, “深層学習”, 初版第12刷, 講談社, 2015, 機械学習プロフェッショナルシリーズ.
3. Sebastian Raschka, Vahid Mirjalili, “[第２版]Python機械学習プログラミング　達人データサイエンティストによる理論と実装”, 第1版第3刷, インプレス, 2019.
4. Diederik Kingma, Jimmy Ba, “Adam : A Method for Stochastic Optimization”, ICLR2015, https://arxiv.org/pdf/1412.6980v8.pdf, 2014.