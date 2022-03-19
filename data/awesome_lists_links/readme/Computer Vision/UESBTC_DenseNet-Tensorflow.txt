# DenseNet-Tensorflow
**Not finished yet**  
Implementing by Tensorflow, it trains and evaluates DenseNet based on CIFAR-100 dataset. 
We have trained the network and release the [weight files](https://drive.google.com/open?id=1lUXFS7Dn7e8YUnVTLsJREvWUJ62yoc0A). The accuracy reaches 81.7% after training for 300 epoches. 
## How to Use
1. Download CIFAR-100 dataset(the Python version) from [here](http://www.cs.toronto.edu/~kriz/cifar.html)
2. Unpack it to  ./DenseNet-Tensorflow 
3. Run ```python3 DenseNet.py```
## Reference
- [Densely connected convolutional networks ZL Gao Huang, KQ Weinberger, L van der Maaten - ARXIV eprint, 2016](https://arxiv.org/abs/1608.06993)
- [Original implementation](https://github.com/liuzhuang13/DenseNet)
