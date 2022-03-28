# face-DCGAN

This is a TensorFlow implementation of [https://arxiv.org/abs/1511.06434](https://arxiv.org/abs/1511.06434).

![DCGAN](./images/DCGAN.png)

## Usage

```
$ python train.py --help

Usage: train.py [OPTIONS]

  Trains the Deep Convolutional Generative Adversarial Network (DCGAN).

  See https://arxiv.org/abs/1511.06434 for more details.

  Args: optional arguments [python train.py --help]

Options:
  -nd, --noise-dim INTEGER   Dimension of noise (1-D Tensor) to feed the
                             generator.
  -glr, --gen-lr FLOAT       Learning rate for minimizing generator loss
                             during training.
  -dlr, --disc-lr FLOAT      Learning rate for minimizing discriminator loss
                             during training.
  -bz, --batch-size INTEGER  Mini batch size to use during training.
  -ne, --num-epochs INTEGER  Number of epochs for training models.
  -se, --save-every INTEGER  Epoch interval to save model checkpoints during
                             training.
  -tb, --tensorboard-vis     Flag for TensorBoard Visualization.
  --help                     Show this message and exit.
```

## Built with

* Python
* TensorFlow
* Numpy

## References

* [Generative Adversarial Networks](https://arxiv.org/pdf/1406.2661)
* [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)

For the Torch implementation of the same, please see [face-DCGAN.torch](https://github.com/pskrunner14/face-DCGAN/tree/master/face-DCGAN.torch)