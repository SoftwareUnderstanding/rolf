Deep Deterministic Policy Gradient using PyTorch
=====

Overview
======
This is my personal and simplified implementation of `Deep Deterministic Policy Gradient` https://arxiv.org/pdf/1509.02971.pdf (DDPG) using `PyTorch` https://github.com/pytorch/pytorch

Dependencies
======
* Python 3.7
* PyTorch 1.0.0 
* `OpenAI Gym` https://github.com/openai/gym

How to run
======
* Clone repository :
```
$ git clone https://github.com/LM095/DDPG-implementation.git 
$ cd DDPG-implementation
```

* Training : results of one environment and its training curves:

	* Pendulum-v0
`
 $ python main.py
`

	<img align="center" src="plot/plot.png">
