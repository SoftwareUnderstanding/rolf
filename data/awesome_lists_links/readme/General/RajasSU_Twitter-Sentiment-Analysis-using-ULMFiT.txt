# Twitter-Sentiment-Analysis-using-ULMFiT
Technologies used: Python, Pandas, Numpy, Sklearn, Nltk, Fastai, Seaborn.

## Problem Description
This repository focuses on implementing the project 'Twitter Sentiment Analysis using ULMFiT' which is consisted of a thorough analysis of the dataset and prediction of Twitter sentiments.
A sentiment analysis job has to be done regarding the problems of each major U.S. airline and contributors. We have first to classify positive, negative, and neutral tweets, and then categorize the negative tweets according to the reasons provided, i.e., "late flight" or "rude service." This problem along with the Dataset is available on Kaggle: https://www.kaggle.com/crowdflower/twitter-airline-sentiment

## Data exploration and processing
Exploratory Data Analysis of the dataset conveyed the missing values in a few columns of the dataset. The columns with more than 90% of the missing values were removed, which included tweet_coord , airline_sentiment_gold, negativereason_gold is missing. The majority of the comments are negative, which means that people are generally dissatisfied with the airline companies' service. The airline's sentiments vary significantly depending on the airline. The most positive is Virgin America, while the most negative is United considering the overall sentiment.

## Model training
We will follow the ULMFiT approach of Howard, and Ruder presented in the paper: https://arxiv.org/pdf/1801.06146.pdf. ULMFit stands for Universal Language Model Fine-tuning; it is an efficient Transfer Learning approach that can be extended to any NLP function and implements language model fine-tuning techniques.
We will also make extensive use of the fastai package as the methods described in the paper are implemented using this package. The paper discusses applying ULMFiT to an IMDB sentiment problem.
ULMFiT consists of three stages:
 a) LM Pre-training: The Language Model (LM) is trained on a general-domain corpus to capture the language's general features in different layers.  Transfer Learning and the ULMFiT method aims to align this pre-trained model with our problem.
b) LM fine-tuning: Wikipedia's language is different from Twitter's, and we need to fine-tune the language model to the dataset. Using discriminative fine-tuning and slanted triangular learning rates (STLR) to learn task-specific features, the full LM is fine-tuned on the target task dataset.
c) Classifier fine-tuning: The classifier is fine-tuned on the target task using gradual unfreezing, discriminative fine-tuning, and STLR to preserve low-level representations and adapt high-level ones. A language model determines the next word if we give the beginning of a sentence. This isn't exactly what we expect. So we're replacing the last layers with several layers of sentiment classification.

## Result
After analyzing all the different learning rates and methods we used, the Accuracy that we got was 0.825 Language modeling can be viewed as the ideal source for NLP, as it encompasses many aspects of language, such as long-term interactions, hierarchical relationships, and sentiments. It provides data in almost unlimited quantities for most domains and languages. As evident from the gradual unfreezing, upon increasing the count of the unfrozen layer per epoch, the validation loss increased, which could result in overfitting the model. The best case we obtained was with two layers unfrozen because, in that case, both the validation loss and the training loss was considerably less.
