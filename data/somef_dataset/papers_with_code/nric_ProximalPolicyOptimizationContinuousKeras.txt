# ProximalPolicyOptimizationContinuousKeras
This is an Tensorflow 2.0 (Keras) implementation of a Open Ai's proximal policy optimization PPO algorithem for continuous action spaces.

Goal was to make it understanable yet not deviate from the original PPO idea: https://arxiv.org/abs/1707.06347 

Part of the code base is from https://github.com/liziniu/RL-PPO-Keras . However, the code there had errors
but mainly it did not use a GAE type reward and no entropy bonus system.

I gave my best to comment the code but I did not include a fundamental lecutre on the logic behind PPO. I highly 
recommend to watch these two videos to undestand what happens:

https://youtu.be/WxQfQW48A4A

https://youtu.be/5P7I-xPq8u8

The most complete explenation and also part of the code (i.e. Memory Class) is from the open ai spinning up project: https://spinningup.openai.com/en/latest/algorithms/ppo.html

I did NOT test this in depth. There might be errors. In a first attempt, the best score was somewhere around -70 for bipedap-walker 
which seems to show some leraning but not great learning.

TODO / Next steps:
1) Try some parameters to find a reasonably quick leraning agent. Currently does not converge or only very slowly.
2) try use tf.distribution to replace maual Probability Density and entropy calculations. 
3) Currently, the two outputs of actor (mu and sigma) are concatenated and then disassembled for the loss. Because the loss depends on both outputs at the same time (mu and sigma). I found this to be the only alternative to writing a custom train fuction with keras.function which seems not to work with TF 2.0 alpha. I should at least try to find a more elegant method.
4) read and implement tf.probability layers independant_normal - does this even make sense here?
