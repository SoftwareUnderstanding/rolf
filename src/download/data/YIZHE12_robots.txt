# ML and DL help robots to navigate

## Background
The goal of this project is to apply machine learning and deep learning methods to predict the surface typo that a robot is on based on time series data recorded by the robot's Inertial Measurement Units (IMU sensors).

## Data
Data can be downloaded in https://www.kaggle.com/c/career-con-2019/data.

The input data has 10 sensor channels of and 128 measurements per time series plus three ID columns:

-row_id: The ID for this row.

-series_id: ID number for the measurement series. Foreign key to y_train/sample_submission.

-measurement_number: Measurement number within the series.

<img src = images/raw_data.png height = 300>

For one example, the data looks like this:

<img src = images/exp.png height = 300>


## Solutions:
This is a time series classification problem. I investigated in two solutions: a machine leanring one and a deep learning one.

### Machine Learning
For the machine learning solution, first, I started by aggregating the time-series data and do feature engineering using the python package [tsfresh](https://tsfresh.readthedocs.io/en/latest/). It automatically generate a list of time series features, such as the following:

<img src = images/features.png height = 700>

For the complete list of avaliable features, please visit: https://tsfresh.readthedocs.io/en/latest/text/list_of_features.html.

I then used a StratifiedKFold to seperate the data to train and validation set. The folds are made by preserving the percentage of samples for each class. I developed an automatic piplline to test different machine learning models, including knn, cart, svm, bayes, random forest, extra trees and gradient boosting models. 

<img src = images/models.png height = 400>

I eventually chose to use ExtraTreesClassifier based on accuracy and speed trade-off. I then used Grid Search Cross Validation for hyperparameters tuning.

<img src = images/tune.png height = 200>

### Deep Learning
I designed a 1D CNN + LSTM classifier for the time series data. I added the batch norm in the input layer so that there is no need to do normalization on the raw data. The 1D CNN network extract features from the time series data in an effective manner. The LSTM layer, which is one time of recurrent neural networks can be used to further capture temporal dynamic features. For more information about RNN, please visit my repo: https://github.com/YIZHE12/math_of_AI/tree/master/SequenceModel, in which I used numpy only to build RNN units. 

<img src = images/LSTM.png height = 200>

One issue of this data set is that it is highly unbalanced. To solve this problem, instead of using cross entropy as the cost function, I adapted the local loss cost function from https://arxiv.org/abs/1708.02002. It is a modulation of cross entropy with a dynamic term based on the confidence of the prediction, forcing the model to learn from the a few hard examples.

<img src = images/focal_loss.png height = 400> (Tsung-Yi Lin, et al)

The final deep learning model acchieved more than 72% accuracy, outperforming the ML model. 









