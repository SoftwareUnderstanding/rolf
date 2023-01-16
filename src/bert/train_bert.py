import numpy as np 
import pandas as pd
from typing import List, Dict
import os
os.system("wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py")
import tensorflow as tf
import tensorflow_hub as hub
from keras.utils import to_categorical
from sklearn import preprocessing
from bert.tokenization import bert_tokenization as tokenization
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
os.system("pip3 install bert-tensorflow==1.0.1")
from bert.tokenization import bert_tokenization as tokenization
import logthis
import sys
from absl import flags
from sklearn.utils import shuffle as sh
from imblearn.under_sampling import RandomUnderSampler
import pickle
import seaborn as sns


logthis.say(f"Version: {tf.__version__}")
logthis.say(f"Eager mode: {tf.executing_eagerly()}")
logthis.say(f"Hub version: {hub.__version__}")
gpu = "available" if tf.config.list_physical_devices("GPU") else "NOT AVAILABLE"
logthis.say(f"GPU is {gpu}")


def get_sampling_strategy(df_train: pd.DataFrame, categories: List[str], cat: str) -> Dict[str, int]:
	df =df_train.drop(df_train[df_train['Label'] == 'General'].index)
	sizes = df.groupby('Label').size()
	indexes = list(sizes.index)
	cat_size = sizes[indexes.index(cat)]
	# If the category is bigger than the sum of the other categories
	#print(df[df[LABEL] != 'Other'].groupby(LABEL).size().sum())
	if cat_size > df.groupby('Label').size().sum() - cat_size:
		cat_size = df.groupby('Label').size().sum() - cat_size
	#other_cat_size = int(cat_size/(len(categories)+1))+1
	other_cat_size = cat_size
	sampling_strategy = {}
	change = 0
	for c in categories:
		if c == cat:
			sampling_strategy[c] = cat_size
		elif c not in categories+['Other']:
			sampling_strategy[c] = 0
		else:
			s = other_cat_size
			sampling_strategy[c] = min(s, sizes[indexes.index(c)])
			change = 0
			if sampling_strategy[c] < s:
				change += s - sampling_strategy[c]
			if sizes[indexes.index(c)] < other_cat_size:
				change += other_cat_size - sizes[indexes.index(c)]
			else: 
				change += 0
	logthis.say(f'Sampling strategy: {str(sampling_strategy)}',)
	return sampling_strategy


def load_data():
    train_data = pd.read_csv('data/train_test_data/readme_negative_samples_preprocessed_train.csv',sep=';')
    train_data = train_data.drop(columns = 'Repo')
    train_data = train_data.drop(train_data[train_data.Label == 'General'].index)

    X = train_data[['Text']]
    y = train_data[['Label']]
    cats = list(train_data['Label'].unique())
    sampling_strategy = get_sampling_strategy(train_data, cats, 'Natural Language Processing')
    rus = RandomUnderSampler(random_state=42, sampling_strategy=sampling_strategy)
    X, y = rus.fit_resample(X, y)
    train_data = pd.concat([X, y], axis=1)
    train_data = sh(train_data)
    print(train_data[['Label']].value_counts())
    
    test_data = pd.read_csv('data/train_test_data/readme_negative_samples_preprocessed_test.csv',sep=';')
    test_data = test_data.drop(columns = 'Repo')
    test_data = test_data.drop(test_data[test_data.Label == 'General'].index)

    return train_data, test_data


def encoding_the_labels(train_data):
    label = preprocessing.LabelEncoder()
    y = label.fit_transform(train_data['Label'])
    print('0, 1, 2, 3, 4, 5: ', label.inverse_transform([0, 1, 2, 3, 4, 5]))
    y = to_categorical(y)
    return label, y


def bert_encode(texts, tokenizer, max_len=512):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for text in texts:
        text = tokenizer.tokenize(text)
        
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len-len(input_sequence)
        
        tokens = tokenizer.convert_tokens_to_ids(input_sequence) + [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len
        
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
        
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)

def build_model(bert_layer, max_len=512):
    input_word_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")
    
    pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    
    clf_output = sequence_output[:, 0, :]
    
    lay = tf.keras.layers.Dense(64, activation='relu')(clf_output)
    lay = tf.keras.layers.Dropout(0.3)(lay)
    lay = tf.keras.layers.Dense(16, activation='relu')(lay)
    lay = tf.keras.layers.Dropout(0.3)(lay)
    out = tf.keras.layers.Dense(6, activation='softmax')(lay)
    
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=2e-5,
        decay_steps=10000,
        decay_rate=0.5)
    
    model = tf.keras.models.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    #model.compile(tf.keras.optimizers.Adam(lr=2e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model


def plot_confusion_matrix(X_test, y_test, model):
    y_pred = model.predict(X_test)
    y_pred = [np.argmax(i) for i in model.predict(X_test)]

    con_mat = tf.math.confusion_matrix(labels=y_test, predictions=y_pred).numpy()

    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
    label_names = list(range(len(con_mat_norm)))

    con_mat_df = pd.DataFrame(con_mat_norm,
                              index=label_names, 
                              columns=label_names)

    figure = plt.figure(figsize=(10, 10))
    sns.heatmap(con_mat_df, cmap=plt.cm.Blues, annot=True)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


if __name__ == "__main__":
    sys.argv=['preserve_unused_tokens=False']
    flags.FLAGS(sys.argv)
    max_len = 300
    
    train_data, test_data = load_data()
    
    
    label, y = encoding_the_labels(train_data)
    #label = preprocessing.LabelEncoder()
    #y = label.fit_transform(train_data['Label'])
    #y = to_categorical(y)

    m_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2'
    # m_url = 'https://tfhub.dev/google/small_bert/bert_uncased_L-2_H-512_A-8/2'
    bert_layer = hub.KerasLayer(m_url, trainable=True)

    tf.gfile = tf.io.gfile
    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
    
    train_input = bert_encode(train_data.Text.values, tokenizer, max_len=max_len)
    test_input = bert_encode(test_data.Text.values, tokenizer, max_len=max_len)
    train_labels = y

    labels = label.classes_
    logthis.say(labels)

    model = build_model(bert_layer, max_len=max_len)
    logthis.say(model.summary())

    checkpoint = tf.keras.callbacks.ModelCheckpoint('/home/u951/u951196/rolf/data/model_1002/model_bert.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
    earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)

    history = model.fit(
        train_input, train_labels,
        validation_split=0.2,
        epochs=200,
        callbacks=[checkpoint, earlystopping],
        batch_size=8,
        verbose=1
    )
    
    with open('/home/u951/u951196/rolf/data/model_1002/history.json', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    logthis.say(history.history.keys())
    
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('/home/u951/u951196/rolf/data/model_1002/bert_accuracy.png')
    plt.clf()
    # summarize history for loss
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('/home/u951/u951196/rolf/data/model_1002/bert_loss.png')
