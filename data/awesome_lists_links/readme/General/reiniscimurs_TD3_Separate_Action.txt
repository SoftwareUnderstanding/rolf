# TD3_Separate_Action
Twin-Delayed Deep Deterministic Policy Gradient network specifically for HalfCheetahBulletEnv-v0 environment using PyTorch. The implementation is based on the standard version of T3D from the Udemy course [Deep Reinforcement Learning 2.0](https://dynatrace.udemy.com/course/deep-reinforcement-learning/learn/lecture/14827394#overview) by Kirill Eremenko and Hadelin de Ponteves. The network is updated for use with the Cheetah agent as it separates the actions for each legs and calculates the action knowing the action of the other leg.

Dependencies: 

* [PyTorch](https://pytorch.org/get-started/locally/)
* [PyBullet](https://github.com/bulletphysics/bullet3)

The replay buffer data structure is written by P. Emami (https://github.com/pemami4911)

Results:

* After 400'000'000 steps:
<p align="center">
    <img width=40% src="https://github.com/reiniscimurs/TD3_Separate_Action/blob/master/Gifs/Cheetah40.gif">
</p>
* After 500'000'000 steps:
<p align="center">
    <img width=40% src="https://github.com/reiniscimurs/TD3_Separate_Action/blob/master/Gifs/Cheetah50.gif">
</p>
* After 1'000'000'000 steps:
<p align="center">
    <img width=40% src="https://github.com/reiniscimurs/TD3_Separate_Action/blob/master/Gifs/Cheetah100.gif">
</p>



The original TD3 paper: https://arxiv.org/abs/1802.09477
