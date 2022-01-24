OVERVIEW:

This project is aimed at detecting sarcasm in a sentence. Detecting sarcasm is essential for effective computing and sentiment analysis.

MODEL SELECTION

ULMFiT (Universal Language Model Fine-tuning for Text Classification), as described in the paper:https://arxiv.org/abs/1801.06146 by Jeremy Howard and Sebastian Ruder, consisting of AWD_LSTM was used for the classification of Sarcastic and Non-sarcastic sentences. 

The pretrained weights for this model was obtained, which was previously trained on WikiText-103 data


TRAINING THE LANGUAGE MODEL

 The encoder was trained with both training as well as test data to increase the vocabulary of the language model.
 
 FastAI’s tokenization method was used to convert the raw data before feeding it to the language model. 
 
Methods like Stemming, Lemmatization and Removing English stopwords are not used because using these leads to a huge loss of contextual information. FastAI’s tokenization uses various tags to preserve the information related to context and position and format of words in the sentences. 

The word “#sarcasm” was removed from the training.csv to prevent overfitting and give better generalized results.


FINAL PREDICTION

With these hyperparameter tunings, the accuracy in our own cross validation reduced to 87.2% but the F1-score in the public leader board for our predictions increased to 0.92278, which shows better generalized performance of the model.

Using dropout rate 0.7 while training the classifier again increased the F1-score to 0.93164 in the public leaderboard

