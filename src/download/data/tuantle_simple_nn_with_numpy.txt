# [Simple NN with Numpy](https://github.com/tuantle/simple_nn_with_numpy)

#### A simple neural network library built with numpy!

This was project was created for educational purposes because the best way to learn and understand the fundamentals of artificial neural network algorithms is to build one up from scratch.

Because this is a pure numpy library, many calculations have to be done manually, such as computing the gradient for backpropagation. For later time, numpy could be replaced with [JAX numpy](https://github.com/google/jax) for the autograd feature.

Below is a list of implemented (and soon to be implemented) features:
- **Linearity & Nonlinearities** - (activation functions)
    - Linear
    - ReLU
    - Leaky ReLU
    - ELU
    - SoftPlus
    - Sigmoid
    - Tanh
    - Swish
        - <img src="https://latex.codecogs.com/svg.latex?&space;\color{blue}f(x)=\frac{xe^x}{1+e^x}=x\varsigma(x)" title="Swish"/>

        - <img src="https://latex.codecogs.com/svg.latex?&space;\color{red}f'(x)=\frac{e^x}{1+e^x}+\frac{xe^x}{1+e^x}\Big(1-\frac{e^x}{1+e^x}\Big)=\varsigma(x)+f(x)\Big(1-\varsigma(x)\Big)" title="Swish Derivative"/>

        [Ref paper - https://arxiv.org/abs/1710.05941](https://arxiv.org/abs/1710.05941)

    <!-- - Algebraic
        - <img src="https://latex.codecogs.com/svg.latex?&space;\color{blue}f(x)=\frac{x}{\sqrt{x^2+1}}" title="Algebraic"/>

        - <img src="https://latex.codecogs.com/svg.latex?&space;\color{red}f'(x)=\frac{1}{(x^2+1)^\frac{3}{2}}" title="Algebraic Derivative"/>

        This activation function has asymptotes of 1 as <img src="https://latex.codecogs.com/svg.latex?&space;x\to\infty" title=""/> and -1 as <img src="https://latex.codecogs.com/svg.latex?&space;x\to-\infty" title=""/>, so it is simlilar to Tanh.
        <p align="center">
            <img width="340" height="340" src="assets/plots/algebraic_nonlinearity.png">
        </p> -->
    [Source](https://github.com/tuantle/simple_nn_with_numpy/blob/master/modules/npcore/layer/gates.py)
- **Regularizations**
    - Batch Normalization - normalize input batch for a specified layer to have mean = 0 and variance = 1 distribution
    - Dropout - dropping out neurons for a specified layer with a given probability
    - Weight - regularize the weights at a specified layer
        - L1 (lasso)
        - L2 (ridge)
        - Combined L1 and L2 (elastic net)
    [Source](https://github.com/tuantle/simple_nn_with_numpy/blob/master/modules/npcore/regularizers.py)
- **Initializations**
    - Zeros
    - Ones
    - Constant
    - Identity
    - RandomBinary
    - RandomOrthonormal
    - RandomUniform
    - RandomNormal
    - GlorotRandomUniform
    - GlorotRandomNormal
    [Source](https://github.com/tuantle/simple_nn_with_numpy/blob/master/modules/npcore/initializers.py)
- **Objectives** - a set of loss functions for minimizing the error between prediction <img src="https://latex.codecogs.com/svg.latex?&space;y" title="truth"/> and truth <img src="https://latex.codecogs.com/svg.latex?&space;\hat{y}" title="truth"/> with gradient descent.
    - Regression Metrics
        - MAE loss (L1 loss)
        - MSE loss (L2 loss)
        - Log-cosh Loss

        <!-- - XTanh Loss
            - <img src="https://latex.codecogs.com/svg.latex?&space;\color{blue}Loss=\frac1{N}\sum_{i=1}^{N}(x_i)\cdot\tanh(x_i)\quad where \quad x_i=y_i-\hat{y_i}" title="XTanh Loss"/>

            - <img src="https://latex.codecogs.com/svg.latex?&space;\color{red}\frac{\partial{Error}}{\partial{y}}=\tanh(x)+x\cdot(1-\tanh^2(x))\quad where \quad x=y-\hat{y}" title="XTanh Error gradient"/>

            Below is the plot of XTanh loss (red) overlaying MSE loss (blue) and MAE loss (green)
            <p align="center">
                <img width="340" height="200" src="assets/plots/xtanh_loss.png">
            </p> -->

        <!-- - Algebraic Loss
            - <img src="https://latex.codecogs.com/svg.latex?&space;\color{blue}Loss=\frac1{N}\sum_{i=1}^{N}\frac{x_i^2}{\sqrt{x_i^2+1}}\quad where \quad x_i=y_i-\hat{y_i}" title="Algebraic Loss"/>

            - <img src="https://latex.codecogs.com/svg.latex?&space;\color{red}\frac{\partial{error}}{\partial{y}}=\frac{x^3+2x}{(x^2+1)^{\frac3{2}}}\quad where \quad x=y-\hat{y}" title="Algebraic Error gradient"/>

            Below is the plot of Algebraic loss (red) overlaying MSE loss (blue) and MAE loss (green)
            <p align="center">
                <img width="340" height="200" src="assets/plots/algebraic_loss.png">
            </p>

            Both Xtanh and Algebraic losses are similar to Log-cosh loss in which they approximate MSE loss for small x and MAE loss for large x. -->

    - Classification Metrics
        - CatergorialCrossentropy Loss (with softmax)
        - CatergorialCrossentropy Accuracy, Recall, Precision, F1-score
        - BinaryCrossentropy Loss (with sigmoid)
        - BinaryCrossentropy Accuracy, Recall, Precision, F1-score
    [Source](https://github.com/tuantle/simple_nn_with_numpy/blob/master/modules/npcore/layer/objectives.py)

- **Obtimizations** - per layer optimation where it is possible to use a combination of optimazation algrorithms in a network.
    - SGD
    - SGD with momentum
    - RMSprop
    - Adam
    [Source](https://github.com/tuantle/simple_nn_with_numpy/blob/master/modules/npcore/optimizers.py)
- **Models** -
    - FeedForward
    - Conv
        - 1D (to be implemented)
        - 2D (to be implemented)
    - Recurrence (to be implemented)
    - Save trained model as a JSON file
    - Load/reload trained model from a JSON file
    [Source](https://github.com/tuantle/simple_nn_with_numpy/blob/master/modules/model)
----

- [Examples](#examples)
- [Installation](#installation)
- [Documentations](#documentations)
- [Change Log](#change-log)
- [License](#license)

----

### Examples

#### Catergorial Classification

In this example, the objective is to perform a simple classification task on a dataset. Below is a plot of 3 group of 2d points (pink, blue, and teal), clustered in a concentric ring patterns, that will be used to as a test dataset for this example.

<p align="center">
    <img width="340" height="300" src="modules/examples/plots/category_classifier_rings_dataset.png">
</p>

First, import 2 moduldes from simple_nn_with_numpy, a **Sequencer** and a **FeedForward**. The job of the sequencer is to serve as a tool for stacking model layers together sequentially. **FeedForward** serves as the base class for the model. It handles model creation, training/predicting, and saving/loading.

```python
import numpy as np

from model.sequencer import Sequencer
from model.nn.feed_forward import FeedForward

import matplotlib.pyplot as plt
from matplotlib.pyplot import animation
from matplotlib.pyplot import cm
import seaborn as sns

# global variables
sample_size = 512  # number of 2d datapoints
dim = 2  # dimensionality
class_size = 3  # number of classes (rings)
training_input_t = np.zeros((sample_size * class_size, dim))  # data matrix (each row = single example)
expected_output_t = np.zeros((sample_size * class_size, class_size))  # data matrix (each row = single example)
expected_output_labels = np.zeros(sample_size * class_size, dtype='uint8')  # class labels

```
For this example, **CategoryClassifierModel** is an inheritance class of **FeedForward**. The next step is to define the model's feed forward layers by creating a class method called **construct**. The purpose of this method is to create and return the model's layers as a sequence.

```python
class CategoryClassifierModel(FeedForward):
    def __init__(self, name):
        super().__init__(name=name)
        self._reports = []  # monitoring data for every epoch during training phase
        self.assign_hook(monitor=self.monitor)  # attach monitoring hook
    def construct(self):
        seq = Sequencer.create(
            'swish',
            size=dim,
            name='input'
        )
        seq = Sequencer.create(
            'swish',
            size=dim * 16,
            name='hidden1'
        )(seq)
        seq.reconfig(
            optim='sgd'
        )
        seq = Sequencer.create(
            'swish',
            size=dim * 12,
            name='hidden2'
        )(seq)
        seq.reconfig(
            optim='adam'
        )
        seq = Sequencer.create(
            'linear',
            size=class_size,
            name='output',
        )(seq)
        seq.reconfig(
            optim='sgd'
        ).reconfig_all(
            weight_init='glorot_random_normal',
            weight_reg='l1l2_elastic_net',
            bias_init='zeros'
        )
        seq.name = 'category_classifier'
        return seq
```

Below is the summary printout of the model's layers sequence created above. Notice that one feature of this library is that different optimizers can be use for each layer. Here, Adam optimizer is use lony for the middle hidden layer with the most parameters and SGD optimizers for the outer layers.

Note that there is an opportunity here to explore whether using a mixture of opimizers rather than one would improve training time & results.

```
Layer	Index	Optim	Type		Shape	   Params
=========================================================
input:
        -------------------------------------------------
	     0                 swish       (*, 2)      2
hidden1:
        -------------------------------------------------
	     1        sgd      link        (2, 32)     66
        -------------------------------------------------
	     2                 swish       (*, 32)     32
hidden2:
        -------------------------------------------------
	     3        adam     link        (32, 24)    800
        -------------------------------------------------
	     4                 swish       (*, 24)     24
output:
        -------------------------------------------------
	     5        sgd      link        (24, 3)     96
        -------------------------------------------------
	     6                 linear      (*, 3)      0
=========================================================
```

Below are some helper class methods to create animation of the training progress and make nice plots of loss/accuracy.

```python
    def monitor(self, report):
        #  this monitor hook is called at the end of every epoch during training
        if report['stage'] == 'learning_and_testing':
            self._reports.append(report)  # save the training report of current epoch

    def on_epoch_end(self, epoch):
        #  save the prediction output at the end of every training epoch for making cool animated plots later on
        (x_min, x_max) = (0, 1)
        (y_min, y_max) = (0, 1)
        (xx, yy) = np.meshgrid(np.arange(x_min, x_max, 0.005), np.arange(y_min, y_max, 0.005))
        testing_input_t = np.c_[xx.ravel(), yy.ravel()]
        predicted_output_t = self.predict(testing_input_t)  # get the current prediction for this epoch
        zz = predicted_output_t.argmax(axis=1).reshape(xx.shape)
        self._training_snapshots.append(zz)

    def plot(self):
        learning_losses = []
        testing_losses = []
        learning_accuracies = []
        testing_accuracies = []
        for report in self._reports:
            evaluated_metric = report['evaluation']['metric']
            learning_losses.append(evaluated_metric['learning']['loss'])
            testing_losses.append(evaluated_metric['testing']['loss'])
            learning_accuracies.append(evaluated_metric['learning']['accuracy'])
            testing_accuracies.append(evaluated_metric['testing']['accuracy'])

        # plotting training loss and accuracy
        figure1 = plt.figure()
        figure1.suptitle('Evaluations')
        plt.subplot(2, 1, 1)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.plot(learning_losses, color='orangered', linewidth=1, linestyle='solid', label='Learning Loss')
        plt.plot(testing_losses, color='salmon', linewidth=1, linestyle='dotted', label='Testing Loss')
        plt.legend(fancybox=True)
        plt.grid()
        plt.subplot(2, 1, 2)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.plot(learning_accuracies, color='deepskyblue', linewidth=1, linestyle='solid', label='Learning Accuracy')
        plt.plot(testing_accuracies, color='aqua', linewidth=1, linestyle='dotted', label='Testing Accuracy')
        plt.legend(fancybox=True)
        plt.grid()

        # plotting prediction result per training epoch
        figure2 = plt.figure()
        figure2.suptitle('Training Results Per Epoch')
        epoch_limit = len(self._training_snapshots)
        (x_min, x_max) = (0, 1)
        (y_min, y_max) = (0, 1)
        (xx, yy) = np.meshgrid(np.arange(x_min, x_max, 0.005), np.arange(y_min, y_max, 0.005))
        imgs = []
        for epoch in range(epoch_limit):
            zz = self._training_snapshots[epoch]
            im = plt.contourf(xx, yy, zz, cmap=cmap_spring)
            imgs.append(im.collections)
        plt.scatter(training_input_t[:, 0], training_input_t[:, 1], c=expected_output_labels, s=5, cmap=cm.get_cmap('cool'))
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.grid()

        anim = animation.ArtistAnimation(figure2, imgs, interval=48, repeat_delay=1000, repeat=True)

        plt.show()

        return anim
```

Finally in the code block below, **generate_rings** method is used to generate concentric rings dataset for training & prediction. And **run_example** is where everything is put together. Here the model is created, trained, and save/load.

```python
# this function generated randomized concentric rings datapoints.
def generate_rings():
    for j in range(class_size):
        ix = range(sample_size * j, sample_size * (j + 1))
        r = 0.125 * (j + 1) + np.random.randn(sample_size) * 0.025
        t = np.linspace(j * 4, (j + 1) * 4, sample_size) + np.random.randn(sample_size) * 0.4  # theta
        training_input_t[ix] = np.c_[r * np.sin(2 * np.pi * t) + 0.5, r * np.cos(2 * np.pi * t) + 0.5]
        expected_output_t[ix] = np.array([1 if j == 0 else 0, 1 if j == 1 else 0, 1 if j == 2 else 0])
        expected_output_labels[ix] = j

def run_example():
    print('Simple Category Classifier Rings Example.')
    # create the model with minimizing softmax crossentropy loss objectiveself.
    # both loss and accuracy are used as training metrics
    model = CategoryClassifierModel(name='CategoryClassifier').setup(objective='softmax_crossentropy_loss',
                                                                     metric=('loss', 'accuracy'))

    # check for previously trained model if available
    if not os.path.isfile('modules/examples/models/category_classifier_rings.json'):
        print(model.summary)
        # generate  concentric rings dataset for training
        generate_rings()
        # start the training...
        model.learn(training_input_t, expected_output_t, epoch_limit=50, batch_size=32, tl_split=0.2, tl_shuffle=True)
        model.save_snapshot('modules/examples/models/', save_as='category_classifier_rings')

        anim = model.plot()
        anim.save('modules/examples/plots/category_classifier_rings.gif', dpi=80, writer='pillow')
    else:
        # load previously trained model
        model.load_snapshot('modules/examples/models/category_classifier_rings.json', overwrite=True)

        print(model.summary)

        (x_min, x_max) = (0, 1)
        (y_min, y_max) = (0, 1)
        (xx, yy) = np.meshgrid(np.arange(x_min, x_max, 0.005), np.arange(y_min, y_max, 0.005))
        testing_input_t = np.c_[xx.ravel(), yy.ravel()]

        # generate new concentric rings dataset for prediction
        generate_rings()

        # start the prediction with trained model...
        predicted_output_t = model.predict(testing_input_t)
        zz = predicted_output_t.argmax(axis=1).reshape(xx.shape)

        figure = plt.figure()
        figure.suptitle('Classification Prediction Results')
        plt.contourf(xx, yy, zz, cmap=cm.get_cmap('spring'), alpha=0.8)
        plt.scatter(training_input_t[:, 0], training_input_t[:, 1], c=expected_output_labels, s=5, cmap=cm.get_cmap('cool'))
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.grid()
        plt.show()

if __name__ == '__main__':
    run_example()
```

Below are the plots of training loss and accuracy. The animated plot shows the training progresses after each training epoch.

[Complete Category Classifier Example Source](https://github.com/tuantle/simple_nn_with_numpy/blob/master/modules/examples/category_classifier.py)

To run this example:
```
cd simple_nn_with_numpy
source bin/activate
python modules/examples/category_classifier.py
```

<p align="center">
    <img src="modules/examples/plots/category_classifier_rings_training.png">
    <img src="modules/examples/plots/category_classifier_rings.gif">
</p>

#### MNIST Recontruction

This is a regression example. MNIST digits are recontructed with simple autoencoder.

[Complete MNIST Reconstruction Example Source](https://github.com/tuantle/simple_nn_with_numpy/blob/master/modules/examples/mnist_reconstruction.py)

To run this example:
```
cd simple_nn_with_numpy
source bin/activate
python modules/examples/mnist_reconstruction.py
```

<p align="center">
    <img src="modules/examples/plots/mnist_reconstruction_training.png">
    <img src="modules/examples/plots/mnist_reconstruction.gif">
</p>

<p align="center">
    <img src="modules/examples/plots/mnist_0_9_recontructed.png">
</p>

### Signal Denoising

This is an other regression example. Using denoising autoencoder to attempt and remove noise and reconstruct the original signal.

[Complete Signal Denoising Example Source](https://github.com/tuantle/simple_nn_with_numpy/blob/master/modules/examples/signal_denoising.py)

To run this example:
```
cd simple_nn_with_numpy
source bin/activate
python modules/examples/signal_denoising.py
```

<p align="center">
    <img src="modules/examples/plots/signal_denoising_training.png">
    <img src="modules/examples/plots/signal_denoising.gif">
</p>

----

### Installation

If using OSX, first install python3 with homebrew. Go here if you don't have homebrew installed(https://brew.sh).

```
brew install python3
```

Use pip3 to install virtualenv and virtualenvwrapper

```
pip3 install virtualenv virtualenvwrapper
```

Add these lines to your .bashrc

```
export WORKON_HOME=$HOME/.virtualenvs
export VIRTUALENVWRAPPER_PYTHON=/usr/local/bin/python3
export VIRTUALENVWRAPPER_VIRTUALENV=/usr/local/bin/virtualenv
source /usr/local/bin/virtualenvwrapper.sh
```

Clone simple_nn_with_numpy project folder from github and activate it with virtualenv:

```
git clone https://github.com/tuantle/simple_nn_with_numpy.git
virtualenv simple_nn_with_numpy
cd simple_nn_with_numpy
source bin/activate
```

In the activated simple_nn_with_numpy dir, install the only required numpy package
```
pip3 install numpy
```

To run the examples, install the following helper packages (torch and tourchvision for the MNIST dataset & utilities):
```
pip3 install pandas, matplotlib, seaborn, torch, torchvision
```

---

### Documentations (WIP)

---

### Change Log
**0.1.1 (8/13/2019)**
```
Notes:
New Features:
Breaking Changes:
Improvements:
    - Replaced string format with f-string.
Bug fixes
    - Fixed a bug in Sequencer load_snapshot method.
```
**0.1.0 (1/30/2019)**
```
Notes:
    - Initial Commit Version!!!
New Features:
Breaking Changes:
Improvements:
Bug fixes:
```

### License

[Prototype](https://github.com/tuantle/simple_nn_with_numpy) is [MIT licensed](./LICENSE).
