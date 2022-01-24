# Actor-Critic Reinforcement Learning

This project intends to provide a documented and extensible
implementation of the A2C and ACKTR algorithms by OpenAI.  

Based on the paper by Wu, Mansimov, Liao, Grosse, and Ba (2017): https://arxiv.org/pdf/1708.05144.pdf  
Original implementation: https://github.com/openai/baselines  

## Documentation

[![Documentation Status](https://readthedocs.org/projects/actor-critic/badge/?version=latest)](https://actor-critic.readthedocs.io/en/latest/?badge=latest)

The API documentation and a quickstart guide can be found on
[Read the Docs](https://actor-critic.readthedocs.io/en/latest/).

## Usage

### Prerequisites

The following dependencies need to be installed
besides [TensorFlow](https://github.com/tensorflow/tensorflow) and [NumPy](https://github.com/numpy/numpy)
(click links for further details):
* [OpenAI gym](https://github.com/openai/gym). Install with:
```
$ pip install gym
```
* [KFAC for TensorFlow](https://github.com/tensorflow/kfac). You need the latest version (0.1.1), which currently is not
hosted on PyPI. Install with:
```
$ pip install git+https://github.com/tensorflow/kfac
```

To use the Atari environments you need:
* [OpenAI atari-py](https://github.com/openai/atari-py). Install with:
```
$ pip install atari-py
```
* [OpenCV for Python](https://github.com/skvark/opencv-python). Install with:
```
$ pip install opencv-python
```

This project is only tested on Linux with Python 3.6.5.

### Examples

Run the following to train an Atari model (see [a2c_acktr.py](actorcritic/examples/atari/a2c_acktr.py) for further
details):
```
$ python -m actorcritic.examples.atari.a2c_acktr
```

If you encounter an InvalidArgumentError 'Received a label value of x which is outside the valid range of [0, x)',
restart the program until it works. This is not intended and hopefully will be fixed in the future.

You can visualize the learning progress by launching
[TensorBoard](https://www.tensorflow.org/guide/summaries_and_tensorboard):
```
$ tensorboard --logdir ./results/summaries
```
