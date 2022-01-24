# tetrisRL
![Python](https://img.shields.io/badge/Python->=3.5-315a81.svg)
![Tensorflow](https://img.shields.io/badge/Tensorflow-1.15.2-c97b2c.svg)
![Keras](https://img.shields.io/badge/Keras-2.3.1-3b8131.svg)

[demo video](https://www.youtube.com/watch?v=jbZ2wG1Pzb4&t=23s)
## Results
play on 7x14 board\
![](https://github.com/SayhoKim/tetrisRL/blob/master/result_1.jpg) ![](https://github.com/SayhoKim/tetrisRL/blob/master/result_2.jpg)
## Quick start
### Installation
1. Clone this repo
```
  $ git clone https://github.com/SayhoKim/tetrisRL.git
```
2. Install Pygame, Keras and Tensorflow
```
  $ cd {Project path}
  $ pip3 install -r requirements.txt
```
### Training
```
  $ python3 main.py
```
### Demo
After editing configuration file (MODELPATH):
```
  $ python3 main.py --demo
```
## Reference
1. Playing Atari with Deep Reinforcement Learning[[arxiv]](https://arxiv.org/abs/1312.5602)
2. Dueling Network Architectures for Deep Reinforcement Learning[[arxiv]](https://arxiv.org/abs/1511.06581)
3. Deep Reinforcement Learning with Double Q-learning[[arxiv]](https://arxiv.org/abs/1509.06461)
4. Prioritized Experience Replay[[arxiv]](https://arxiv.org/abs/1511.05952)
