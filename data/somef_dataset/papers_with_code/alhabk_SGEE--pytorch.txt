# --Self Generate Expert Experience/SGEE 
This code combine DDPG Algorithm and Behavior Clone methods,which integrate off and on-policy training process.
After one episode on-policy train, algorithm generate expert samples with current parameters and feed the off-policy train.
For it can produce expert experient by itself, so we call it SGEE.
The implementation of DDPG refer to sweetice's code.>>https://github.com/sweetice/Deep-reinforcement-learning-with-pytorch

Reference
CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING>>https://arxiv.org/abs/1509.02971
Self Lmitation Learning>>https://arxiv.org/abs/1806.05635

