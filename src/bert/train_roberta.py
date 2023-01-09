import sys
import os
import numpy as np
import regex as re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
import math
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras.backend as K
import tokenizers
from keras.utils import to_categorical
from transformers import RobertaTokenizer, TFRobertaModel
from collections import Counter
import warnings
from absl import flags


MODEL_NAME = 'roberta-base'
MAX_LEN = 256


def load_data():
    train_data = pd.read_csv('data/train_test_data/readme_new_preprocessed_train.csv',sep=';')
    train_data = train_data.drop(columns = 'Repo')
    train_data = train_data.drop(train_data[train_data.Label == 'General'].index)

    test_data = pd.read_csv('data/train_test_data/readme_new_preprocessed_test.csv',sep=';')
    test_data = test_data.drop(columns = 'Repo')
    test_data = test_data.drop(test_data[test_data.Label == 'General'].index)

    return train_data, test_data


def roberta_encode(texts, tokenizer):
    ct = len(texts)
    input_ids = np.ones((ct, MAX_LEN), dtype='int32')
    attention_mask = np.zeros((ct, MAX_LEN), dtype='int32')
    token_type_ids = np.zeros((ct, MAX_LEN), dtype='int32') # Not used in text classification

    for k, text in enumerate(texts):
        # Tokenize
        tok_text = tokenizer.tokenize(text)
        
        # Truncate and convert tokens to numerical IDs
        enc_text = tokenizer.convert_tokens_to_ids(tok_text[:(MAX_LEN-2)])
        
        input_length = len(enc_text) + 2
        input_length = input_length if input_length < MAX_LEN else MAX_LEN
        
        # Add tokens [CLS] and [SEP] at the beginning and the end
        input_ids[k,:input_length] = np.asarray([0] + enc_text + [2], dtype='int32')
        
        # Set to 1s in the attention input
        attention_mask[k,:input_length] = 1

    return {
        'input_word_ids': input_ids,
        'input_mask': attention_mask,
        'input_type_ids': token_type_ids
    }
    

def get_strategy():
    try:
        # TPU detection. No parameters necessary if TPU_NAME environment variable is set (always set in Kaggle)
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
        print('Running on TPU ', tpu.master())
    except ValueError:
        # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
        strategy = tf.distribute.get_strategy()

    print('Number of replicas:', strategy.num_replicas_in_sync)
    return strategy


def build_model(n_categories, strategy):
    with strategy.scope():
        input_word_ids = tf.keras.Input(shape=(MAX_LEN,), dtype=tf.int32, name='input_word_ids')
        input_mask = tf.keras.Input(shape=(MAX_LEN,), dtype=tf.int32, name='input_mask')
        input_type_ids = tf.keras.Input(shape=(MAX_LEN,), dtype=tf.int32, name='input_type_ids')

        # Import RoBERTa model from HuggingFace
        roberta_model = TFRobertaModel.from_pretrained(MODEL_NAME)
        x = roberta_model(input_word_ids, attention_mask=input_mask, token_type_ids=input_type_ids)

        # Huggingface transformers have multiple outputs, embeddings are the first one,
        # so let's slice out the first position
        x = x[0]

        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dense(n_categories, activation='softmax')(x)

        model = tf.keras.Model(inputs=[input_word_ids, input_mask, input_type_ids], outputs=x)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=1e-5),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

        return model


if __name__ == "__main__":
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
    session = tf.compat.v1.InteractiveSession(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
    sys.argv=['preserve_unused_tokens=False']
    flags.FLAGS(sys.argv)
    
    strategy = get_strategy()
    
    df, test_data = load_data()
    
    X_data = df[['Text']].to_numpy().reshape(-1)
    y_data = df[['Label']].to_numpy().reshape(-1)
    
    label = preprocessing.LabelEncoder()
    df['Label'] = label.fit_transform(df['Label'])
    df['Label'] = to_categorical(df['Label'])
    
    categories = df[['Label']].values.reshape(-1)
    n_categories = len(categories)
    
    category_to_id = {}
    category_to_name = {}

    for index, c in enumerate(y_data):
        if c in category_to_id:
            category_id = category_to_id[c]
        else:
            category_id = len(category_to_id)
            category_to_id[c] = category_id
            category_to_name[category_id] = c
        
        y_data[index] = category_id

    # Display dictionary
    category_to_name
    
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=777) 
    
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    
    X_train = roberta_encode(X_train, tokenizer)
    X_test = roberta_encode(X_test, tokenizer)

    y_train = np.asarray(y_train, dtype='int32')
    y_test = np.asarray(y_test, dtype='int32')
    
    
    with strategy.scope():
        model = build_model(n_categories, strategy)
        model.summary()
        print('Training...')
        checkpoint = tf.keras.callbacks.ModelCheckpoint('/home/u951/u951196/rolf/data/model_1002/roberta_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
        earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)
        history = model.fit(X_train,
                            y_train,
                            epochs=10,
                            batch_size=32       ,
                            callbacks=[checkpoint, earlystopping],
                            verbose=1,
                            validation_data=(X_test, y_test))
        
        # summarize history for accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig('/home/u951/u951196/rolf/data/model_1002/roberta_accuracy.png')
        # summarize history for loss
        
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig('/home/u951/u951196/rolf/data/model_1002/roberta_loss.png')
    
    