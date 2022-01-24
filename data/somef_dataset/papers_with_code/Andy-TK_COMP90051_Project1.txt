### COMP90051 Project 1: Authorship Attribution with Limited Text on Twitter

This is the Project 1 for COMP90051 (Statistical Machine Learning) from the University of Melbourne.

### 1. What is the task? 
Authorship attribution is a common task in Natural Language Processing (NLP) applications, such as academic plagiarism detection and potential terrorist suspects identification on social media. As for the traditional author classification task, the training dataset usually includes the entire corpus of the author’s published work, which contains a large number of examples of standard sentences that might reflect the writing style of the author. However, when it comes to the limited text on social media like Twitter, it brings some challenging problems, such as informal expressions, a huge number of labels, unbalanced dataset and extremely limited information related to identity.

<img src="https://github.com/Andy-TK/COMP90051_Project1/blob/master/Result/kaggle.png" alt="Kaggle" width="70%">

In this project, the task is to predict authors of test tweets from among a very large number of authors found in training tweets, which comes from an in-class [Kaggle Competition](https://www.kaggle.com/c/whodunnit/leaderboard). Our works include data preprocessing, feature engineering, model selection and ensemble models etc. For more details, please check the [project specifications](https://github.com/Andy-TK/COMP90051_Project1/blob/master/Project%20specifications.pdf) and [project report](https://github.com/Andy-TK/COMP90051_Project1/blob/master/Project%20Report%20Team%2052.pdf).

### 2. Data
The `Data` folder contains both original data and processed data.
#### 2.1. Original Data
`train_tweets.txt`
> _The original training dataset which contains 328932 tweets posted by 9297 users._

<img src="https://github.com/Andy-TK/COMP90051_Project1/blob/master/Data/01_original_train.png" alt="original training data" width="70%">

`test_tweets_unlabeled.txt`
> _The original test dataset which contains 35437 tweets posted by the same user group in the training dataset._

<img src="https://github.com/Andy-TK/COMP90051_Project1/blob/master/Data/02_original_test.png" alt="original training data" width="70%">

#### 2.2. Processed Data
The `preprocess.py` in the `Code` folder transfered the original data into processed data. For example:

<img src="https://github.com/Andy-TK/COMP90051_Project1/blob/master/Data/03_processed_data.png" alt="original training data" width="70%">

`all_clean_data.csv`
> _The entire processed training dataset which contains 328932 tweets posted by 9297 users._

`test_clean_data.csv`
> _The entire processed test dataset which contains 35437 tweets posted by the same user group in the training dataset._

`train.csv`
> _The random 9/10 processed training dataset used for partial training dataset._

`train.csv`
> _The random 1/10 processed training dataset used for partial test dataset._

### 3. Code
#### 3.1. Data Preprcessing and Feature Engineering
`preprocess.py`
> _is used for data preprocessing including removing non-English characters (e.g. emoticons and punctuations) and stopwords, as well as word tokenization and lemmatization based on [nltk](https://www.nltk.org/) package. Also, it provides some distribution plots for data based on [matplotlib](https://matplotlib.org/) package._

<img src="https://github.com/Andy-TK/COMP90051_Project1/blob/master/Result/Figure_2_numplot.png" alt="numplot" width="50%"><img src="https://github.com/Andy-TK/COMP90051_Project1/blob/master/Result/Figure_1_boxplot.png" alt="boxplot" width="50%">

Before entering data into the models，using **TF-IDF** to transfer clean tweets text into a vector or matrix. This process is implemented by `CountVectorizer` and `TfidfTransformer` modules from [_scikit-learn_](https://scikit-learn.org/stable/) package.

#### 3.2. Model Selection
Five machine learning/deep learning models based on [_scikit-learn_](https://scikit-learn.org/stable/) and [_Keras_](https://keras.io/) are implemented in this part, including the **_Multinomial Naive Bayes_**, **_KNN_**, **_Multiple Logistic Regression_**, **_Linear SVC_** and **_LSTM_**.

* `nb.py` - Multinomial Naive Bayes Model.
* `knn1.py` - KNN Model.
* `mlr.py` - Multiple Logistic Regression Model.
* `svc.py` - Linear Support Vector Classifier Model.
* `lstm.py` - LSTM Model.

#### 3.3. Ensemble Learning
`ensemble.py`
> _Ensemble learning is a powerful technique to increase accuracy on a most of machine learning tasks. In this project, we try a simple ensemble approach called weighted voting to avoid overfitting and improve performance. The basic thought of this method is quite simple. For each prediction from the results of different models, we give them a weight corresponding to their individual accuracy in the previous stage. If the predicted labels of two models are the same, we just add their weight together. Then we select the prediction with highest weight as the final prediction._

Considering the individual performance of the previous models, we try three different combinations: 
* linearSVC + MultinomialNB + KNN(K=1)
* linearSVC + MultinomialNB + MLR
* linearSVC + MultinomialNB + MLR + KNN(K=1).

### 4. Future Works
Due to the limitation of time, we have some ideas which might be worthy but have not yet to try:
* Using [SMOTE](https://arxiv.org/pdf/1106.1813.pdf) algorithm to deal with the unbalanced training dataset.
* Hyper-parameter optimization based on grid search technique.
* Adjusting the weight of penalty item. Giving large value when the prediction for minority class is wrong and small value when the  prediction for majority class is wrong.
* Some more complicated but powerful ensemble learning methods which can be found [here](https://mlwave.com/kaggle-ensembling-guide/).
