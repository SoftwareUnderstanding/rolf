# CS 236 Entropy Regularized Conditional GANs

*Example usage*

Running conditional gan (train_cond.py) with regularization term 0.0002

`$ python main.py --dataset cifar10 --model presgan --lambda 0.0002`

Files: hmc.py -> HMC Sampling, train_cond.py -> Main conditional gan training with nets_cond.py architecture.

Entropy on MNIST 
![Mnist](https://raw.githubusercontent.com/evazhang612/gan-results/master/mnist0.0/presgan_mnist_fake_epoch_041.png)

Entropy Saturation on CelebA (0.0002)
![CelebA](https://github.com/evazhang612/gan-results/blob/master/hardcodedceleba/presgan_celeba_fake_epoch_015.png)

Entropy Saturation on Cifar-10 0.0002)
![Saturation](https://raw.githubusercontent.com/evazhang612/gan-results/master/cifar10_00005/presgan_cifar10_fake_epoch_077.png)

Modulated Lambda on Entropy Saturation on Cifar 10 - 0.0002
![Adapted](https://raw.githubusercontent.com/evazhang612/gan-results/master/00002png/presgan_cifar10_fake_epoch_040.png)
