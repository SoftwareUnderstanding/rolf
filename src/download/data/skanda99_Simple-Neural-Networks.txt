### SIMPLE NEURAL NETWORKS

Simple Neural Networks (SiNN) is a library for building sequential dense neural networks. Support for standard ingredients for a neural network are provided like activation functions, loss functions, weights/bias initializers, optimizers (Adam, SGD, BGD) and some necessary operations like feed-forward and back-propagation. Any of the above can be customized according to requirement and used as just a reference to the functions are passed as arguments in most of the cases. Each of the optimizer can calculate Receiver Operating characteristics (ROC) for training set.

The library uses numpy and random frameworks in the background. A sample MNIST digit classification file has been put up for reference.

## Activation functions
    - Linear
    - Sigmoid
    - Softmax
    - ReLU
    - Tangent Hyperbolic

## Loss functions
    - L2 / MSE
    - L1 / MAE
    - MBE
    - Binary cross entropy
    - Cross entropy
    - Huber loss
    - Square Epsilon Hinge loss

## Weights and Bias initializers
    - Zero
    - Standard Uniform distribution
    - Standard Normal distribution
    - Xavier (normal)
    - He (normal)

## Classification Metrics
    - Precision
    - Recall / Sensitivity / True Positive rate
    - Specificity
    - F1 score

## Optimizers
    - Adam
    - SGD
    - BGD


## References
    - https://numpy.org/doc/stable/
    - https://docs.python.org/3/library/random.html
    - https://www.deeplearning.ai/ai-notes/initialization/
    - https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79
    - https://arxiv.org/abs/1412.6980
    - https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5
