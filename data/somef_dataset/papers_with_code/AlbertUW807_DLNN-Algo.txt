# DLNN-Algo
〽️ Deep Learning & Neural Networks Projects 〽️

### Install Numpy
```
$ install numpy
```

### Projects 

#### [Logistic Regression](https://github.com/AlbertUW807/DLNN/tree/master/Logistic%20Regression)
  - Implemented an Image Recognition Algorithm that recognizes cats with 67% accuracy!
  - Used a logistic regression model.
  
#### [Deep Learning Model](https://github.com/AlbertUW807/DLNN/tree/master/Deep%20Learning%20Model)
  - Implemented an Image Recognition Algorithm that recognizes cats with 80% accuracy!
  - Used a 2-layer neural network (LINEAR->RELU->LINEAR->SIGMOID) 
            and an L-layer deep neural network ([LINEAR->RELU]*(L-1)->LINEAR->SIGMOID).
  - Trained the model as a 4-layer neural network.

#### [Model Initialization](https://github.com/AlbertUW807/DLNN/tree/master/Model%20Initialization)
  - Implemented different initialization methods to see their impact on model performance (3-Layer).
  - Zero Initialization -> Fails to break symmetry (all parameters to 0).
  - Random Initialization -> Breaks symmetry, more efficient models.
  - He Initialization -> Xavier Initialization without scaling factor, recommended for layers with ReLU activation.

#### [Regularization Methods](https://github.com/AlbertUW807/DLNN/tree/master/Regularization%20Methods)
  - Used a deep learning model to determine which player does the goalkeeper have to pass to from a noisy dataset.
  - Implemented a model in regularization and dropout mode to see how different regularization methods affect it.
  - Better accuracy on the training set over the test set without regularization.

#### [Gradient Check](https://github.com/AlbertUW807/DLNN/tree/master/Gradient%20Check)
  - Implemented a One-Dimensional and an N-Dimensional Gradient Check.
  - Used the difference formula to check the backward propogation.
  - Able to identify which parameter's gradient was calculated incorrectly.


#### [Optimization](https://github.com/AlbertUW807/DLNN/tree/master/Optimization)
  - Used mini-batch gradient descent.
  - How momentum affects performance of a model.
  - Adam and RMS prop.
  - Training the 3-Layer Neural Network
    - Mini-batch Gradient Descent
    - Mini-batch Momentum
    - Mini-batch Adam
  - Adam References: https://arxiv.org/pdf/1412.6980.pdf.
