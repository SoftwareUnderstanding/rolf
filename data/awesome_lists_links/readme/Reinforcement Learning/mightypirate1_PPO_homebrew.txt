# PPO_homebrew

This is a reimplementation of Proximal Policy Optimization PPO (originally https://arxiv.org/abs/1707.06347).

It runs with any OpenAI gym environment.

The implementation is as far as I can tell faithful to the original publication as, and works phenomenally well on simple applications. Due to the limited hardware available to me, I can not tell how the performance of this implementation compares to that of the original when it comes to harder environments such as Atari etc.

## Installation:
Install python (3.6.3) with TensorFlow (1.8.0) and NumPy (1.16). Numbers in parenthesis were the vesion used for development, but may not be required. Pull the repo, and run if from that folder.

## Train:
To train an agent on environment E for S steps and save as X:
```
python3 runner.py --train --env E --steps S --name X
```
> If the environment E is an Atari environment, you might also add "--atari" to the command line to get the hyperparameters for that experiment from the original paper. An arsenal of wrappers is optionally applied "DeepMind-style". To control which, find the class "wrap_atari" in wrappers.py and uncomment the ones you like.

## Test:
To test an agent saved as X on environment E for S steps:
```
python3 runner.py --test --env E --steps S --name X
```
> If the environment E is an Atari environment and you used the "--atari" flag for training, you should also use it here! Make sure the wrappers enabled are the same as for training (else your results are probably gonna be worse, or there will be errors).

## Comments, questions or corrections?
In case of any of the above, drop me a line on yfflan at gmail dot com :-)
