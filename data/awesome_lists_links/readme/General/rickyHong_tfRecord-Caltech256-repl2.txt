# ResnetCNN
Repository for creating a deep residual network in tensorflow
https://gist.github.com/danijar/c7ec9a30052127c7a1ad169eeb83f159 (ref)

### Resnet:
based on paper, [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf) He et al.2015

The Resnet is similar to a convolutional neural network with a 'shortcut'. A shortcut is a path that goes through a group of convolutional layers, directly allowing the input to pass to the output. This creates a stablizing effect similar to an anchor, by supporting the parameters across deep networks, preventing degradation. Resnets allows for generalization to increase with layer depth.

Ran on [Cifar10](https://www.cs.toronto.edu/~kriz/cifar.html) and [Caltech256](http://www.vision.caltech.edu/Image_Datasets/Caltech256/) datasets

### Files
The files in this repository include:
- caltech256_bin.py 
	- converts caltech256 images to tfrecord files for faster reading
- caltech256_input.py
	- input functions to batch images and labels
- cifar10_input.py
	- input functions to batch images and labels
- model.py
	- the resnet model, can use caltech256 input or cifar10 input
- train.py
	- trains the model
- eval.py
	- runs test batch evalutation for model
- specifimage.py
	- evaluates a specific image in folder
	
### Results:
Ran on Ubuntu with Geforce GTX 1070 GPU
Successful
- 12 hours of training on cifar for 40470 iterations achieves 80% test accuracy
	- could have been ran for longer, to see if converges more.
- 17 hours of training on Caltech256 for 61100 iterations
	- training accuracy: 60%
	- testing accuracy: 0.8%
	- result of overfitting
	
### Further Improvements
Further improvements to the code:
- create validation set
- better preprocessing for greater invariance 
- greater amount of training data
- apply build to other platforms
- eliminate useless code
- add more error systems for better security and debugging
- introduce batch normalization mean and variance as variable which are retained in the model during evaluation

###### References:
@article{DBLP:journals/corr/HeZRS15,
  author    = {Kaiming He and
               Xiangyu Zhang and
               Shaoqing Ren and
               Jian Sun},
  title     = {Deep Residual Learning for Image Recognition},
  journal   = {CoRR},
  volume    = {abs/1512.03385},
  year      = {2015},
  url       = {http://arxiv.org/abs/1512.03385},
  timestamp = {Wed, 07 Jun 2017 14:41:17 +0200},
  biburl    = {http://dblp.uni-trier.de/rec/bib/journals/corr/HeZRS15},
  bibsource = {dblp computer science bibliography, http://dblp.org}
}


CIFAR10: https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf

Caltech: Griffin, G. Holub, AD. Perona, P. The Caltech 256. Caltech Technical Report
