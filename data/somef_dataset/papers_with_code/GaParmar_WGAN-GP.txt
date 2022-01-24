# WGAN-GP
pytorch implementation of wgan-gp

# Usage
- `python3 train.py <exp_name>`
- Config file must be provided at `./configs/<exp_name>.py`
- experiment logs saved in `./EXP_LOGS/log_<exp_name>.txt`

# Sample Results
## MNIST (100 epochs, batch_size=512)
- Training command to reproduce: `python3 train.py defaults_mnist`
- ![IMNIST](results/mnist_100_512.png)

## CIFAR (250 epochs, batch_size=64)
- Training command to reproduce: `python3 train.py defaults_cifar10`
- ![I_CIFAR10](results/cifar10_250_64.png)

## References
- paper: https://arxiv.org/pdf/1704.00028.pdf
- https://github.com/caogang/
