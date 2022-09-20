from pathlib import Path
from keras.layers import Dropout, Dense, Embedding, LSTM, Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
import logthis
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
logging.basicConfig(level=logging.INFO)


TEXT = "Text"
LABEL = "Label"
CV_splits = 5


def prepare_model_input(X_train, X_test,MAX_NB_WORDS=500,MAX_SEQUENCE_LENGTH=50):
    np.random.seed(7)
    logthis.say(X_train.shape)
    logthis.say(X_train.head())
    logthis.say(X_test.shape)
    logthis.say(X_test.head())
    text = np.concatenate((X_train, X_test), axis=0)
    text = np.array(text)
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(text)
    # pickle.dump(tokenizer, open('text_tokenizer.pkl', 'wb'))
    # Uncomment above line to save the tokenizer as .pkl file 
    sequences = tokenizer.texts_to_sequences(text)
    word_index = tokenizer.word_index
    text = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    logthis.say('Found %s unique tokens.' % len(word_index))
    indices = np.arange(text.shape[0])
    # np.random.shuffle(indices)
    text = text[indices]
    logthis.say(text.shape)
    X_train_Glove = text[0:len(X_train), ]
    X_test_Glove = text[len(X_train):, ]
    embeddings_dict = {}
    f = open("glove.6B.50d.txt", encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype='float32')
        except:
            pass
        embeddings_dict[word] = coefs
    f.close()
    logthis.say('Total %s word vectors.' % len(embeddings_dict))
    return (X_train_Glove, X_test_Glove, word_index, embeddings_dict)
## Check function


#x_train_sample = ["Lorem Ipsum is simply dummy text of the logthis.saying and typesetting industry", "It is a long established fact that a reader will be distracted by the readable content of a page when looking at its layout"]
#x_test_sample = ["I’m creating a macro and need some text for testing purposes", "I’m designing a document and don’t want to get bogged down in what the text actually says"]
#X_train_Glove_s, X_test_Glove_s, word_index_s, embeddings_dict_s = prepare_model_input(df_train['Text'].values.tolist(), 
#                                                                                       df_test['Text'].values.tolist(), 100, 20)
#logthis.say("\n X_train_Glove_s \n ", X_train_Glove_s)
#logthis.say("\n X_test_Glove_s \n ", X_test_Glove_s)
#logthis.say("\n Word index of the word testing is : ", word_index_s["testing"])
#logthis.say("\n Embedding for the word want \n \n", embeddings_dict_s["want"])


def build_bilstm(word_index, embeddings_dict, nclasses,  MAX_SEQUENCE_LENGTH=50, EMBEDDING_DIM=50, dropout=0.5, hidden_layer = 3, lstm_node = 32):
    # Initialize a sequebtial model
    model = Sequential()
    # Make the embedding matrix using the embedding_dict
    embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_dict.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            if len(embedding_matrix[i]) != len(embedding_vector):
                logthis.say("could not broadcast input array from shape", str(len(embedding_matrix[i])),
                      "into shape", str(len(embedding_vector)), " Please make sure your"
                                                                " EMBEDDING_DIM is equal to embedding_vector file ,GloVe,")
                exit(1)
            embedding_matrix[i] = embedding_vector
            
    # Add embedding layer
    model.add(Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True))
    # Add hidden layers 
    for i in range(0,hidden_layer):
        # Add a bidirectional lstm layer
        model.add(Bidirectional(LSTM(lstm_node, return_sequences=True, recurrent_dropout=0.2)))
        # Add a dropout layer after each lstm layer
        model.add(Dropout(dropout))
    model.add(Bidirectional(LSTM(lstm_node, recurrent_dropout=0.2)))
    model.add(Dropout(dropout))
    # Add the fully connected layer with 256 nurons and relu activation
    model.add(Dense(256, activation='relu'))
    # Add the output layer with softmax activation since we have 2 classes
    model.add(Dense(nclasses, activation='softmax'))
    # Compile the model using sparse_categorical_crossentropy
    model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    return model


def get_eval_report(labels, preds):
    mcc = matthews_corrcoef(labels, preds)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    precision = (tp)/(tp+fp)
    recall = (tp)/(tp+fn)
    f1 = (2*(precision*recall))/(precision+recall)
    return {
        "mcc": mcc,
        "true positive": tp,
        "true negative": tn,
        "false positive": fp,
        "false negative": fn,
        "pricision" : precision,
        "recall" : recall,
        "F1" : f1,
        "accuracy": (tp+tn)/(tp+tn+fp+fn)
    }

def compute_metrics(labels, preds):
    assert len(preds) == len(labels)
    return get_eval_report(labels, preds)

def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string], '')
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()


def filter_dataframe(df, cat):
	count = 0
	for ind, row in df.iterrows():
		if cat != str(row[LABEL]):
			count += 1
			row[LABEL] = 'Other'
		
	logthis.say(f'{cat} filtered {count} rows in training dataset')

def get_sampling_strategy(df_train, cat):
	sizes = df_train.groupby(LABEL).size()
	indexes = list(sizes.index)
	cat_size = sizes[indexes.index(cat)]
	other_cat_size = int(cat_size/(len(df_train[LABEL].unique())-2))+1
	sampling_stratgy = {}
	for c in df_train[LABEL].unique():
		if c == cat:
			sampling_stratgy[c] = cat_size
		elif c == 'General':
			sampling_stratgy[c] = 0
		else:
			sampling_stratgy[c] = min(other_cat_size, sizes[indexes.index(c)])
	logthis.say(f'Sampling strategy: {sampling_stratgy}')
	return sampling_stratgy

categories = ["Sequential", "Natural Language Processing", "Audio", "Computer Vision", "Graphs", "Reinforcement Learning"]

def prepare_train_test(cat):
    df_train = pd.read_csv('../../data/train_test_data/readme_new_preprocessed_train.csv', sep=';')
    df_test = pd.read_csv('../../data/train_test_data/readme_new_preprocessed_test.csv', sep=';')
    df_train.drop( df_train[ df_train['Text'] == "" ].index , inplace=True)
    df_test.drop( df_test[ df_test['Text'] == "" ].index , inplace=True)
    df_train.drop_duplicates(subset=['Text'], inplace=True, keep=False)
    
    df_train.drop_duplicates(subset=['Text'], inplace=True, keep=False)
    df_train.drop( df_train[ df_train[TEXT] == "" ].index , inplace=True)
    df_train.drop( df_train[ df_train[LABEL] == "General" ].index , inplace=True)
    df_test.drop( df_test[ df_test[LABEL] == "General" ].index , inplace=True)
    df_test.drop( df_test[ df_test[TEXT] == "" ].index , inplace=True)
    
    logthis.say(f'Train test split starts for {cat=} category')
    df_train = df_train.drop(columns = 'Repo')
    x_train = df_train[TEXT]
    y_train = df_train[LABEL]
    x_test = df_test[TEXT]
    y_test = df_test[LABEL]
    
    undersample = RandomUnderSampler(sampling_strategy=get_sampling_strategy(df_train, cat))
    x_train, y_train = undersample.fit_resample(x_train.to_frame(TEXT), y_train)
    x_train = x_train[TEXT]

    y_train = y_train.to_frame(LABEL)
    filter_dataframe(y_train, cat)
    y_test = y_test.to_frame(LABEL)
    filter_dataframe(y_test, cat)
    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    return x_train, x_test, y_train, y_test

if __name__ == "__main__":
    for i, cat in enumerate(categories):
        
        x_train, x_test, y_train, y_test = prepare_train_test(cat)
        logthis.say("Preparing model input ...")
        X_train_Glove, X_test_Glove, word_index, embeddings_dict = prepare_model_input(x_train, x_test)
        logthis.say("Done!")
        logthis.say("Building Model!")
        model = build_bilstm(word_index, embeddings_dict, 2)
        model.summary()
        logthis.say("Training model!")
        history = model.fit(X_train_Glove, y_train,
                                validation_data=(X_test_Glove,y_test),
                                epochs=100,
                                batch_size=2,
                                verbose=1)
        logthis.say("Saving model!")
        path = Path('../../results/models/keras/')
        path.mkdir(parents=True, exist_ok=True)
        model.save(path / cat)

        plot_graphs(history, 'accuracy')
        plot_graphs(history, 'loss')

        logthis.say(f'\n Evaluating Model for cat: {cat}... \n')
        predicted = model.predict(X_test_Glove)
        logthis.say(predicted)
        predicted=np.argmax(predicted,axis=1)
        logthis.say(predicted)
        logthis.say(metrics.classification_report(y_test, predicted))
        logthis.say("\n")
        result = compute_metrics(y_test, predicted)
        for key in (result.keys()):
            logthis.say(f"  {key} = {str(result[key])}")
