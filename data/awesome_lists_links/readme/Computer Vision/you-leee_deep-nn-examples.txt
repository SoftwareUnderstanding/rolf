


# deep-nn-examples

This repository contains toy examples of shallow and deep neural networks along with convolutional and recurrent neural networks.
It also contains a (not optimal) implementation of base neural networks.
Have fun! :)

## Usage
You can either run the examples from command line or from an IDE. The below examples are for command line usage.

### Setup
First set up the environment by running the setup.py file for installation. It will download all the necessary packages to run the examples.

```
> python setup.py install
```

### Run example

```
> python NeuralNetworks/logisticregression_vs_shallownn.py
```

## List of examples:

This is the list of finished examples.. others will follow!

#### Neaural Networks and Regression
* Simple linear regression
`NeuralNetworks/simple_linear_regression.py`

  Plain and simple implementation of linear regression aming to demonstrate how you can approximate data points, that are close to a linear function (in this example y = 2\*x + 4).

Cost            |  Fitted vs values
:-------------------------:|:-------------------------:
<img src="https://raw.githubusercontent.com/you-leee/deep-nn-examples/master/docs/images/slr_curve.png" width="450">|<img src="https://raw.githubusercontent.com/you-leee/deep-nn-examples/master/docs/images/slr_fitted.png" width="450">
 
* Document classification with word embedding
`NeuralNetworks/doc_classification_apple.py`

  An example on how to learn word embeddings using a neural network. The training data contains text from both Apple Inc. and the apple fruit and the goal is to categorize new text into one of these classes. There is a lot of room for improvement, like getting more training data, filtering stop words better or restricting the vocabulary... Feel free to play around!
 
   `Predicting for sample sentence about the apple fruit: 
  "The world crop of apples averages more than 60 million metric tons a year. Of the American crop, more than half is normally used as fresh fruit."
Prediction: 0.8536878228187561, actual value: 1`

* Logistic regression vs shallow neural networks
`NeuralNetworks/logisticregression_vs_shallownn.py`
    
    In this example, the aim is to classify a linearly NOT separable dataset. You can see, how much better you can do with a neural network with 1 hidden layer vs a simply logistic regression model. It also demonstartes the increase of accuracy, when we increase the size of the hidden layer.
  
Original dataset|  Fit with logistic regression| Fit with different number of layers
:-----------------:|:-----------------:|:------------------------:
<img src="https://raw.githubusercontent.com/you-leee/deep-nn-examples/master/docs/images/log_vs_sh_origdata.png" width="290">|<img src="https://raw.githubusercontent.com/you-leee/deep-nn-examples/master/docs/images/log_vs_sh_slr_fitted.png" width="290">|<img src="https://raw.githubusercontent.com/you-leee/deep-nn-examples/master/docs/images/log_vs_sh_layer_num_fits.png" width="470">

* Shallow neural networks vs "deeper" neural networks
`NeuralNetworks/shallownn_vs_deepnn.py`

   Classic image binary classification problem: cat vs non-cat. Two neural networks are trained for the same number of iterations, but one with 3 hidden layers and the other with only 1. You can observe, that despite, that the simpler model can reach the same train accuracy, on the test set, there is a significant difference.
`2 layer model:	train accuracy: 100 % test accuracy: 70 %`
`4 layer model: train accuracy: 100.0 % test accuracy: 80.0 %`

2 layer network cost|  4 layer network cost| Prediction
:-----------------:|:-----------------:|:------------------------:
<img src="https://raw.githubusercontent.com/you-leee/deep-nn-examples/master/docs/images/shallow_vs_deep_2layer_cost.png" width="350">|<img src="https://raw.githubusercontent.com/you-leee/deep-nn-examples/master/docs/images/shallow_vs_deep_4layer_cost.png" width="350">|<img src="https://raw.githubusercontent.com/you-leee/deep-nn-examples/master/docs/images/shallow_vs_deep_pred.png" width="350">

* Hand (number) sign classification with tensorflow
`NeuralNetworks/tf_sign_classification.py`

   A 1 hidden layer neural network is used to classify hand signs to numbers (0-9). It is an example on how to implement a simple model using tensorflow, instead of coding the backpropagation/optimization yourself.

Cost function| Prediction
:-----------------:|:-----------------:
<img src="https://raw.githubusercontent.com/you-leee/deep-nn-examples/master/docs/images/sign_class_cost.png" width="450">|<img src="https://raw.githubusercontent.com/you-leee/deep-nn-examples/master/docs/images/sign_class_pred.png" width="450">
---
#### Convolutional Neural Networks
* Hand sign classification with convolutional networks
`ConvolutionalNeuralNetworks/cnn_sign_classification.py`

   This demo uses convolutional (and pooling) layers to address the same problem as in the example above ("Hand (number) sign classification with tensorflow" ). The main advantage of using convolutional layers on images is, that you have much less parameters as with a fully connected layer. For example: If the images are only of size 32x32x3 (32 wide, 32 high, 3 color channels), a single fully-connected neuron in a first hidden layer would have 32\*32\*3 = 3072 weights, whereas a convolutional layer with one 4x4 filter has only 4\*4\*3 = 48.

Architecture|
:-----------------:
<img src="https://raw.githubusercontent.com/you-leee/deep-nn-examples/master/docs/images/cnn_arch.png" width="800">|

* The RESNET50
`ConvolutionalNeuralNetworks/resnet_mnist.py`

   In this example, the famous mnist hand written digit dataset is used to train the ResNet50 network, which uses residual blocks. It is a bit "overkill" to use such a big network for this task, but the goal here is to learn a bit about deep residual networks. So what is a residual network? The main idea is, that to "tweek" the mathematical formula with an identity function, such as: f(x) + x = f(x) + id(x) = y. Identity connections enable the layers to learn incremental, or residual representations. The layers can start as the identity function and gradually transform to be more complex. This significantly helps deeper networks in the training process, since the gradient signal vanishes with increasing network depth, but the identity connections in ResNets propagate the gradient throughout the model.

Accurately classified examples| Misclassified examples
:-----------------:|:-----------------:
<img src="https://raw.githubusercontent.com/you-leee/deep-nn-examples/master/docs/images/rnn_acc_classified.png" width="450">|<img src="https://raw.githubusercontent.com/you-leee/deep-nn-examples/master/docs/images/rnn_misclassified.png" width="400">

* YOLO (you only look once)
`ConvolutionalNeuralNetworks/yolo_car_detection.py`
   YOLO is a new approach of object detection with a great performance on real-time. A single neural network predicts bounding boxes and class probabilities directly from full images in one evaluation. The classic model can process 45 frames per second, that's why it's popularity.
   
 Architecture|
:-----------------:|
<img src="https://raw.githubusercontent.com/you-leee/deep-nn-examples/master/docs/images/yolo_architecture.png" width="850">|

   The model has 2 main steps: 
   - Detection: A single convolutional network simultaneously predicts multiple bounding boxes and class probabilities for those boxes.
<img src="https://raw.githubusercontent.com/you-leee/deep-nn-examples/master/docs/images/yolo_detection.png" width="490">|

   - Filtering: Boxes with less probabilty, than the threshhold are disregarded. On the remaining boxes , a simple non-max surpression function prunes away boxes that have high intersection-over-union (IOU) overlap with maximum probability box.


## References
- Python setup: https://docs.python.org/3/distutils/setupscript.html
- Tensorflow: https://www.tensorflow.org
- Deep learning course: https://www.coursera.org/specializations/deep-learning
- Intuitive explonation of ConvNets: https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/
- ConvNet CIFAR-10: https://cs.stanford.edu/people/karpathy/convnetjs/demo/cifar10.html
- Word embeddings: https://www.analyticsvidhya.com/blog/2017/06/word-embeddings-count-word2veec/
- Deep Residual Networks: https://github.com/KaimingHe/deep-residual-networks
- YOLO original paper: https://arxiv.org/pdf/1506.02640.pdf
