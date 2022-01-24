# Model-Optimizer_Implementation

## 前言

深度學習的進展日新月異，許多的資料科學家都在努力的為了找到更好的 Model、效率更好的 Optimizer 在做努力。
這個 Repo 主要針對ㄧ些新模型、新優化器的實現進行實作，並且應證相關論文的實驗結果，當然也可提供工作、專案上能夠有更好效能的泛化模型。

## 模型 Models

1. ShuffleNet v2

    * 相關論文 :  https://arxiv.org/abs/1807.11164  
    Ma, Ningning, Xiangyu Zhang, Hai-Tao Zheng and Jian Sun.  
    “*ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design.*”  
    ArXiv abs/1807.11164 (2018): n. pag.
    
   * ShuffleNet v2 for keras :   
   https://github.com/opconty/keras-shufflenetV2


## 優化器 Optimizers

1. Lookahead

    * 相關論文 :  https://arxiv.org/abs/1907.08610  
    R. Zhang, Michael & Lucas, James & Hinton, Geoffrey & Ba, Jimmy. (2019).   
    *Lookahead Optimizer: k steps forward, 1 step back.* 
    
    * Lookahead for keras :  
    https://github.com/bojone/keras_lookahead

2. Rectified Adam

    * 相關論文 :  https://arxiv.org/abs/1908.03265  
    Liu, Liyuan & Jiang, Haoming & He, Pengcheng & Chen, Weizhu & Liu, Xiaodong & Gao, Jianfeng & Han, Jiawei. (2019).   
    *On the Variance of the Adaptive Learning Rate and Beyond.* 
    
    * RAdam for keras (1) :  
    https://github.com/CyberZHG/keras-radam
    * RAdam for keras (2) :  
    https://github.com/titu1994/keras_rectified_adam