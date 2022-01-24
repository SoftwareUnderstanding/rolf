# Wasserstein-GAN

PyTorch implementation of [Wasserstein GAN](https://arxiv.org/abs/1701.07875) by Martin Arjovsky, et al. on the [MNIST](https://yann.lecun.com/exdb/mnist/) dataset.

<div align='center'>
   <img src="img/progress.gif" alt="progress" align='center' width='240'/>
</div>

### Loss and Training
The network uses Earth Mover's Distance instead of Jensen-Shannon Divergence to compare probability distributions. 

<div align='center'>
   <img src="img/equation.png" alt="minimax" align='center' width='600'/>
</div>


I modeled the generator and critic both using Multi Layer Perceptrons to verify some of the paper's claims. The log(D(x)) trick from the original GAN paper is used while training. The hyperparameters used are as described in the paper. After a few hundred epochs, this was the loss curve.


<div align='center'>
   <img src="img/loss.png" alt="loss_curve" align='center' width="500"/>
</div>


### References
> 1. **Martin Arjovsky, et al.** *Wasserstein GAN.* [[arxiv](https://arxiv.org/abs/1701.07875)]
> 2. **Yann LeCun, et al.** *MNIST Database of Handwritten Digits* [[webpage](https://yann.lecun.com/exdb/mnist/)]

