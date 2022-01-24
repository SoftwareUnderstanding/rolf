### Implementation of LAMB in Tensorflow: https://arxiv.org/pdf/1904.00962.pdf


I am starting with the Tensorflow code for Adam: https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/python/training/adam.py

The only function to change is the _apply_dense function. 

Since TF code does the actual math in C++ (hence the mysterious training_ops.apply_adam function returned), 
I started with the Adam implementation here for __apply_dense: https://github.com/angetato/Optimizers-for-Tensorflow/blob/master/tf_utils/Adam.py

Still working on some issues with this implementation...use with caution.
