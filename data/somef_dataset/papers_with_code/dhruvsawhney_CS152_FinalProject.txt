# CS152_FinalProject (Neural Networks)

## Project: Sentiment Analysis on IMDB movie reviews using Recurrent Neural Network <br />

Prediction: 1/0 for a positive/negative movie review sentiment <br />
Data Source: Kaggle IMDB movie review (https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/data) <br />

### Model overview:

#### Model 1. Sentiment analysis using word embedding (notebooks/RNN_with_word_embedding.ipynb)

  An RNN (architecture details in notebook) using the Keras deep-learning framework that uses a word-embedding layer
  (GloVe embedding) that takes the raw text and feeds dense real-value vectors as input to the RNN.
 
  
#### Model 2. Sentiment analysis using Transfer Learning (notebooks/TransferLearning.ipynb)
  
  This approach is modelled after the paper (https://arxiv.org/abs/1801.06146) and implementation (https://github.com/fastai/fastai/blob/master/courses/dl1/lesson4-imdb.ipynb).
  
  The goal is to recreate this paper to perform transfer learning for sentiment classification, by first creating a language model based on a corpus of text.
  Finally, this model is tuned to perform sentiment classification. Techniques used to train the LM for sentiment analysis can be found in the paper and notebook.
  
  The corpus used for training the language model is another IMDB dataset. Then, transfer learning is performed using the Data Source (cited above).


### Results:

####  Model 1

  Accuracy (trained for 3 epoch cycles): 68 % (without dropout), 73 % (with dropout)

####  Model 2

  Accruacy (trained for 3 epoch cycles): 82 % (with dropout)


### Log of time spent:
  
  Worked from November 5, 2018 to December 9, 2018
  
  Spent roughly 20-25 hours for entire project. Includes time researching, design and time to train each of the 2 models (45   minutes each). <br/>
  
  Most time was spent researching on NLP deep-learning techniques that culminated in both models.
  
  




