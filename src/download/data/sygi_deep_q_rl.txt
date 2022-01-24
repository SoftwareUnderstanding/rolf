# Introduction 

This repository is a fork of the Nathan Sprague implementation of the deep
Q-learning algorithm described in:

[Playing Atari with Deep Reinforcement Learning](http://arxiv.org/abs/1312.5602)
Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis
Antonoglou, Daan Wierstra, Martin Riedmiller

and 

Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning." Nature 518.7540 (2015): 529-533.

We use the DQN algorithm to learn the strategies for Atari games using the RAM state of the machine.

# Dependencies

* A reasonably modern NVIDIA GPU
* OpenCV
* [Theano](http://deeplearning.net/software/theano/) ([https://github.com/Theano/Theano](https://github.com/Theano/Theano))
* [Lasagne](http://lasagne.readthedocs.org/en/latest/) ([https://github.com/Lasagne/Lasagne](https://github.com/Lasagne/Lasagne)
* [Pylearn2](http://deeplearning.net/software/pylearn2/) ([https://github.com/lisa-lab/pylearn2](https://github.com/lisa-lab/pylearn2))
* [Arcade Learning Environment](http://www.arcadelearningenvironment.org/) ([https://github.com/mgbellemare/Arcade-Learning-Environment](https://github.com/mgbellemare/Arcade-Learning-Environment))

The script `dep_script.sh` can be used to install all dependencies under Ubuntu.


# Running
We've done a number of experiments with models that use RAM state. They don't fully share the code, so we split them in branches. To re-run them, you can use our scripts, which are located in the main directory of the repository.

## Network types

- just_ram - network that takes only RAM as inputs, passes it through 2 ReLU layers with 128 nodes each and scales the output to the appropriate size
- big_ram - the analogous network, but with 4 hidden layers
- mixed_ram - network taking both ram and screen as an input
- big_mixed_ram - deeper version of mixed_ram
- ram_dropout - the just_ram with applied dropout to all the layers except the output
- big_dropout - the big_ram network with dropout

## Frame skip
Evaluation of a model using a different frame skip:
```
./frameskip.sh <rom name> <network type> <frameskip>, e.g:
./frameskip.sh breakout just_ram 8
```

## Dropout
We added dropout to the two ram-only networks. You can run it as:
```
./dropout.sh <rom name> ram_dropout
OR
./dropout <rom name> big_dropout
```

`ram_dropout` is a network with two dense hidden layers, `big_dropout` with 4.

## Weight-decay
You can try the models with l2-regularization using:
```
./weight-decay.sh <rom name> <network type>, e.g:
./weight-decay.sh breakout big_ram
```

## Decreasing learning-rate
The models with learning rate decreased to $0.001$ can be run as:
```
./learningrate.sh <rom name> <network type>, e.g:
./learningrate.sh breakout big_ram
```

## Roms
You need to put roms in the `roms` subdirectory. Their names should be spelled with lowercase letters, e.g. `breakout.bin`.

# See Also

* https://github.com/spragunr/deep_q_rl

  Original Nathan Sprague implementation of DQN.

* https://sites.google.com/a/deepmind.com/dqn

  This is the code DeepMind used for the Nature paper.  The license
  only permits the code to be used for "evaluating and reviewing" the
  claims made in the paper.

* https://github.com/muupan/dqn-in-the-caffe

  Working Caffe-based implementation.  (I haven't tried it, but there
  is a video of the agent playing Pong successfully.)

* https://github.com/kristjankorjus/Replicating-DeepMind

  Defunct?  As far as I know, this package was never fully functional.  The project is described here: 
  http://robohub.org/artificial-general-intelligence-that-plays-atari-video-games-how-did-deepmind-do-it/

* https://github.com/brian473/neural_rl

  This is an almost-working implementation developed during Spring
  2014 by my student Brian Brown.  I haven't reused his code, but
  Brian and I worked together to puzzle through some of the blank
  areas of the original paper.

