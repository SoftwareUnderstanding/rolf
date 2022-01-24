# Computer Vision Ferrari Detective
## Overview

This is a completed computer vision project. The objective of this project was to train a convolutional neural network to detect a Ferrari Testarosssa out of a repository of vehichles.

## Execution
This method of using a pre-trained model is called transfer learning. Transfer learning gives everyone access to robust models, making machine learning and artificial intelligence widely accessible without access to expensive training computers. 

From work performed in the deep learning space, research has shown that convolutional networks can be substantially deeper, more accurate, and efficient to train if they contain shorter connections between layers close to the input and those close to the output. This is the reason I chose the the densenet pre-trained model. After embedding a pre-trained model, in the next step I had to create a fully connected layer or hidden layer. These nodes will act as an input layer to these fully-connected layers. Within these layers is where my network will be generating percentages for deciding between a Ferrari Tesarossa or not. Since the output I am looking for only has 2 options, I created a binary classification model compared to categorical model. (The closer the prediction percentage is to 0%, the more it believes the car is a Ferarri Testarossa)

‘Dense’ is the function to add a fully connected layer. These values will always be between the number of input nodes and the output nodes, but choosing the most optimal number of nodes can be achieved only through experimental tries. It took many iterations to realize to get my model to be effective, it required one layer of 128 output nodes, followed by another layer of 1 output node. 

Finally I had to decide whether or not to use a softmax or sigmoid activation function. Activation functions of a node defines the output of that node, given a set of inputs. I chose sigmoid because it can be preferred over softmax if there are isn’t multiple classes and each input doesn’t have to belong to exactly one class.

After training my model, my last epoch had a 99.3% accuracy. 


## Challenges

**Incorrect Labels:** The first data set I used contained about 8100 cars in the training and validation, which I though was a great lead. Unfortunately the labels were incorrect so I was forced to find another dataset. Though unfortunate, this was a useful challenge because it taught me about how to properly label my data for the training to be effective. The way I did so was by creating two directories titled TEST and TRAIN, each with two more directories within titled 'Ferarri Testarossa' and 'Not a Ferrari Testarossa'.

**Low Accuracy:** Initially my accuracy was 60% I was able to dramatically improve my accuracy to about 99.3% by changing the amount of output nodes within my dense fully connected layer. Since changes as these are not consistent and vary accross different models, I predict it worked because for binary classification, it may help if your last layers are closer to one, since the final activation layer will only have one node.

**High Accuracy, but low Validation Accuracy:** Another issue was after changing the amount of output nodes, my accuracy improved, but unfortunately my model still was unable to properly identify Ferrari Testarossas. I identified that my use of regularizers, an overfitting prevention technique, was too powerful. I subsequently removed the regularizers, and was able to gain even higher accuracy and my model was now finally able to now properly predict Testarossas, without a need for preprocessing the data.


https://arxiv.org/pdf/1608.06993.pdf
