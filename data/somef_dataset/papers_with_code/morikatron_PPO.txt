# PPO
Proximal Policy Optimization and Generalized Advantage Estimation implementation with Tensorflow2  
This implementation only supports CartPole environment(OpenAI gym).  

このリポジトリは強化学習アルゴリズムProximal Policy Optimization及びGeneralized Advantage EstimationをTensorflow2で実装したものです。（学習環境はCartPoleにのみ対応しています。）  
PPOについて解説したブログはこちらになります（2020年6月29日10:00より公開）  
https://tech.morikatron.ai/entry/2020/06/29/100000

## Relevant Papers
 - Proximal Policy Optimization Algorithms, Schulman et al. 2017  
https://arxiv.org/abs/1707.06347
 - High-Dimensional Continuous Control Using Generalized Advantage Estimation, Schulman et al. 2016  
https://arxiv.org/abs/1506.02438

## Requirements
 - Python3
 - tensorflow2
 - gym
 - tqdm

## Usage
  - clone this repo
 ```
 $ git clone https://github.com/morikatron/PPO.git
 ```
  - change directory and run 
 ```
 $ cd PPO
 $ python algo/run.py
 ```
 ## Performance Example
 ![CartPole-v1](https://github.com/morikatron/PPO/blob/master/ppo_result.png)

