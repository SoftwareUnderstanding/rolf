import os
import argparse
import sys
import json
import pandas as pd
import numpy as np
import nltk
import time
from nltk import word_tokenize
from pandas.core.indexes.base import Index
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pickle
import random

#nltk.download('punkt')

def sample_together(n, X, y):
    rows = random.sample(np.arange(0,len(X.index)).tolist(),n)
    return X.iloc[rows,], y.iloc[rows,]

def undersample(X, y, under = 'Natural Language Processing'):
    y_min = y[y['Label'] == under]
    y_max = y[y['Label'] != under]
    X_min = X.filter(y_min.index,axis = 0)
    X_max = X.filter(y_max.index,axis = 0)

    X_under, y_under = sample_together(len(y_min.index), X_max, y_max)

    X = pd.concat([X_under, X_min])
    y = pd.concat([y_under, y_min])
    print(X.head())
    print(y.head())
    return X, y


#Preprocess data from Papers with code
df_train = pd.read_csv('dataset/train.csv', sep=';')
print(df_train.shape)
df_train.drop_duplicates(inplace=True)
print(df_train.shape)
df_train_x = df_train['Text']
df_train_y = df_train['Label']


#Plot the label's distribution
unique, counts = np.unique(df_train_y, return_counts=True)
plt.bar(unique, counts, 1)
plt.title('Class Frequency')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.show()


# Generate data using somef
"""

data_somef = {'Text':[], 'Label':[]}

for index, i in df_train.iterrows():
#for j in range(1):
    repo = i['Repo']
    label = i['Label']
    print(repo)
    os.system("somef describe -r {} -o tmp.json -p -t 0.8".format(repo))
    #https://github.com/Sanyuan-Chen/RecAdam
    #time.sleep(10.0)
    res = json.load(open('tmp.json'))
    text = ''
    if 'description' in res:
        #values = [val['excerpt'].replace('\n', ' ') for val in res[key]]
        for i in res['description']:
            if i['technique'] == "Header extraction":
                #print(i['excerpt'])
                text += i['excerpt'].replace('\n', ' ').replace(',', ' ')
                text += ' '
    if 'installation' in res:
        for i in res['installation']:
            #print(i['excerpt'])
            text += i['excerpt'].replace('\n', ' ').replace(',', ' ')
            text += ' '
    if 'usage' in res:
        for i in res['usage']:
            if i['technique'] == "Header extraction":
                text += i['excerpt'].replace('\n', ' ').replace(',', ' ')
                text += ' '
    #print(text)
    data_somef['Text'].append(text)
    data_somef['Label'].append(label)
    os.remove("tmp.json") 
    with open("res.txt", "a", encoding='utf8') as f:
        f.write("Text: {}, \n Label: {} \n\n".format(text, label))
    #results.write("Text: {}, \nLabel: {}".format(text, label))
    print("Text: {}, \nLabel: {}".format(text, label))

    

df_somef = pd.DataFrame(data_somef)
df_somef.to_csv('dataset/train_somef.csv', sep=';', index=False)
print(df_somef.head())

"""


#Train test split

ngram_counter = CountVectorizer(analyzer="word",
                                   tokenizer=word_tokenize,
                                   max_features=None,
                                   lowercase=False)
df_train_x = ngram_counter.fit_transform(df_train_x)

#df_train_x, df_train_y = undersample(df_train_x, df_train_y)

# build the feature matrices


X_train, X_test, y_train, y_test = train_test_split(df_train_x, df_train_y, test_size=0.10, random_state=42, stratify=df_train_y)
#y_train = y_train.values.ravel()
#X_train = X_train.values.ravel()

#X_test  = ngram_counter.transform(X_test)


#Plot the label's distribution
unique, counts = np.unique(y_train, return_counts=True)
plt.bar(unique, counts, 1)
plt.title('Class Frequency')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.show()

unique, counts = np.unique(y_test, return_counts=True)
plt.bar(unique, counts, 1)
plt.title('Class Frequency')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.show()

#Model
clf = Pipeline([
    ('vectorizer', CountVectorizer(analyzer="word",
                                   tokenizer=word_tokenize,
                                   max_features=None,
                                   lowercase=False)),
    ('classifier', SVC())
])




# train the classifier
classifier = LinearSVC()
clf = classifier.fit(X_train, y_train)

y_test = clf.predict(X_test)
print(y_test)

clf.fit(X_train, y_train)


# save the model to disk
filename = 'model_balanced_undersample2.sav'
pickle.dump(clf, open(filename, 'wb'))


