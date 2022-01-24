# Surface cracks detection in pavements

Capstone project for the Springboard Machine Learning Engineering Career Track

## Introduction

Computer vision is used for surface defects inspection in multiple fields, like manufacturing and civil engineering. In this project, the problem of detecting cracks in a concrete surface has been tackled using a deep learning model.

## Data

SDNET2018 is an annotated dataset of concrete images with and without cracks from bridge decks, walls and pavements (https://www.kaggle.com/aniruddhsharma/structural-defects-network-concrete-crack-images). The pavements subset, which includes 2600 positive images (with crack) and 21700 negative images (without crack), has been used to train and test this model. First, the data have been divided into three sets, namely train (80%), validation (10%) and test (10%), then, only for the train subset, new images with cracks have been created and saved to balance the two classes. An example of data augmentation is reported in notebooks/DataAugmentation.ipynb  

## Model

#### Convolutional Neural Network architecture

The architecture used for this project is based on convolutional neural network (CNN) and it is inspired by the many architectures reported in literature, especially VGG16. It consists in the sequence of 5 CNN blocks, the first three blocks have a convolutional layer, followed by a batch normalization, relu activation function and a max pool layer. The last two blocks have two consecutives convolutional+batch normalization+ReLu layers before the max pool. This approach allows to increase the effective receptive field, limiting the number of trainable parameters and accelerating the training.
The stack of convolutional layers is followed by two fully connected layers, with a final softmax activation function that performs the binary classification.

L2 and dropout regularization have been used in the first fully connected layer to limit the overfitting. Dropout has been used only after the last batch normalization layer to avoid variance shift (https://arxiv.org/pdf/1801.05134.pdf). Experimentation showed that adding L2 regularization to the convolutional layers does not improve the performances.

#### Model training

The network has been trained with a GPU P5000, using Adam optimizer and binary crossentropy loss function. The learning rate has been decreased exponentially, from an initial value of 1e-3, with a decay step of 35 and decay rate of 0.92.

After 10 epochs (batches of 128 images), the train loss is stable around 0.105 and the validation loss is around 0.199, which correspond to a ROC AUC of 0.992 and 0.918 respectively.  

The training of the model is saved in notebook/ModelTraining.

#### Hyperparameter Tuning

#### Model Testing

The model has been tested on the dedicated test set, that showed a loss of 0.183, similar to the validation set. To convert the probability to class labels, an optimal threshold has been extracted from the validation set through the  expression: optimal_threshold = argmin(TruePositiveRate - (1-FalsePositiveRate)) and used on both validation and test set. The optimal threshold results in the following metrics:

| | ROC_AUC | Precision | Recall | f1_score | f2_score |
|---|---|---|---|---|---|
| Validation | 0.918 | 0.390 | 0.842 | 0.533 | 0.684|
|Test | 0.921 | 0.366 | 0.824 | 0.507 | 0.659 |

#### Examples of correct classification

Here few examples of corrected classifiaction on test data for both positive and negative examples. The title of each image indicate the actual class and the probability that the image contains a crack. Note that the optimal threshold evaluated on validation data is equal to 0.08 (i.e. p<0.078 --> Non-cracked, p>0.08 --> Cracked)

![True Positive Examples](https://github.com/simo-bat/Crack_detection/tree/master/TruePositive.png?raw=true)

##### Example of missclassification 

Here few examples of missclassified images for both positive and negative examples. 

False Positive
![False Positive Examples](https://github.com/simo-bat/Crack_detection/tree/master/FalsePositive.png?raw=true)

False Negative
![False Negative Examples](FalseNegative.png?raw=true)

False positive examples show common features like stripes and granules. False negative examples show common features like very small and shallow cracks and pothholes that looks like granules/stains.

In general, many images were manually analized and im multiple cases it was very hard to classify them also for a person.  

## Repository description

notebooks/ contains an example of data augmentation (DataAugmentation), the training of the model (ModelTraining), the hyperparameters tuning and the relative impact on the loss function (HyperparametersDependences) and the testing of the trained model (ModelTesting)

app/ contains all the files to run the application: the trained model with the weights, the Flask application, the Dockerfile and the requirements file

test_images/ contains few images from the test subset that can be used to test the app

## Test the app

1) build the docker image: docker build -t crack_api .

2) create the image and start it: docker run -p 5000:5000 crack_api

3) go to http://0.0.0.0:5000/ and test the app 