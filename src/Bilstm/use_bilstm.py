from keras.layers import Dropout, Dense, Embedding, LSTM, Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from sklearn.metrics import matthews_corrcoef, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.utils import shuffle
import numpy as np
import pickle
import matplotlib.pyplot as plt
import warnings
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import RandomUnderSampler
from bi_lstm2_2 import prepare_train_test, prepare_model_input, compute_metrics
import logthis

cat = 'Sequential'
model: Sequential = load_model(f'../../results/models/keras/{cat}')
x_train, x_test, y_train, y_test = prepare_train_test(cat)
logthis.say("Preparing model input ...")
X_train_Glove, X_test_Glove, word_index, embeddings_dict = prepare_model_input(x_train, x_test)
logthis.say(f'\n Evaluating Model for {cat=}... \n')
predicted = model.predict(X_test_Glove)
logthis.say(predicted)
predicted=np.argmax(predicted,axis=1)
logthis.say(predicted)
logthis.say(metrics.classification_report(y_test, predicted))
logthis.say("\n")
result = compute_metrics(y_test, predicted)
for key in (result.keys()):
	logthis.say(f"  {key} = {str(result[key])}")