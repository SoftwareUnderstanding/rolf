# ResNet_MiniProject: ResNet in Tensorflow for CIFAR-10

**ResNet** 


---


The formulation for my notebook is based on He et al.'s [paper](https://arxiv.org/abs/1512.03385).

General design decisions:

*   Full pre-activation residual block (see: [residual block variants](https://miro.medium.com/max/1400/1*M5NIelQC33eN6KjwZRccoQ.png))

*   Perform 1x1 convolution instead of padding input (projection shortcut) for matching output size 

Formulation based on [paper](https://arxiv.org/abs/1512.03385): 
*   1st layer is 3x3 convolutions 
*   Stack of 6n layers with 3x3 convolutions on feature maps of size {32,16,8} respectively (n = number of residual blocks) 
*   Ends with global average pooling, a FC layer, and softmax

**Dataset** 

About [CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz):

*   Image size = 32x32
*   Label Dimension = 10 (classes)
*   60000 images total - 50000 training, 10000 testing 
*   5 training batches, 1 test batch

**Training Logs** 

Google drive links to logs files:
*   [test_2_steps=10000](https://drive.google.com/drive/folders/1mFRgQPsh8C44Z1YlMq2MkNMxaBxw9jwc?usp=sharing)
*   [test_3_steps=20000](https://drive.google.com/drive/folders/133I5Y6YUzWwuf1BsG81Myk_U4y-_Xjq1?usp=sharing)

---
**Justifications/General Thoughts**:

Why did I use projections? 
> The [paper](https://arxiv.org/abs/1512.03385) talks about a few options for increasing dimensions, one being zero-padding, and another being the projection shortcut. They mention that projections are slightly better than zero-padding, but that projections usually mean greater time complexity and model size. I decided to use 1x1 convolutions (projection) anyways since my model is not very deep and after experimenting with zero-padding. 

Why didn't I use any dropout? 
> Based on my readings so far, I think dropout would only have a small effect on overfit, since batch normalization (which is heavily used in ResNet) performs regularization already and contributes to overfit reduction, and that's why dropout is mostly excluded in ResNets. (However, after more digging, I found that doing last-layer dropout on CIFAR10 could bring improvements for DenseNet (https://arxiv.org/pdf/1801.05134.pdf).)

Why is ResNet preferred for deeper models? 
> The main advantage to ResNet is that it addresses the vanishing gradient problem, which is basically when a neural net's weights become very small, and then backpropogation leads to these weights becoming 'vanishingly' small so that effectively, the net stops training. The other side to this problem is exploding gradients, which was mentioned in the first page of the [paper](https://arxiv.org/abs/1512.03385). The other problem with deeper networks is degration (when the accuracy gets saturated and then degrades as the network gets deeper). The intuition behind ResNet and why it solves these problems is that with the residual blocks, you have identity mapping in the worst case instead of a vanishingly small gradient. In essence, the identity mapping occurs when the weight of a previous activation becomes 0 (which usually leads to vanishing/exploding gradient problem, but in this it is mapped to identity).
