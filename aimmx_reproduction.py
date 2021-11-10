import os
import argparse
import sys
import json
import pandas as pd
import numpy as np
import nltk
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

#nltk.download('punkt')

#Preprocess data from Papers with code
data = {'Text':[], 'Label':[]}
methods = json.load(open("methods.json"))
urls = []
labels = ['Computer Vision', 'Natural Language Processing']

for i in methods:
    #if len(i['collections']) == 0 or not i['code_snippet_url'] or i['collections'][0]['area'] not in labels:
    if len(i['collections']) == 0 or i['collections'][0]['area'] not in labels:
        continue
    #urls.append(i['code_snippet_url'].split('#')[0].rsplit('/', 3)[0])
    text = ''
    for key, value in i.items():
        if key != 'collections':
            text += str(value) + ' '
    data['Text'].append(text)
    data['Label'].append(i['collections'][0]['area'])
#print(json.dumps(data, indent=4))

df_train = pd.DataFrame(data)
df_train_x = df_train['Text']
df_train_y = df_train['Label']
#print(df_train.head())

#Plot the label's distribution
unique, counts = np.unique(df_train_y, return_counts=True)
plt.bar(unique, counts, 1)
plt.title('Class Frequency')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.show()


"""
keys_to_use = ['documentation', 'description']
data_test = {'Text':[]}

for i in range(10):
    print(urls[i])
    os.system("somef describe -r {} -o tmp.json -t 0.8".format(urls[i]))
    res = json.load(open('tmp.json'))
    text = ''
    for key in keys_to_use:
        if key in res:
            values = [val['excerpt'].replace('\n', ' ') for val in res[key]]
            text += ' '.join(values) + ' '
    data_test['Text'].append(text)
    #print(text)

df_test = pd.DataFrame(data_test)
#print(df_test.head())
"""

#Train test split
X_train, X_test, y_train, y_test = train_test_split(df_train_x, df_train_y, test_size=0.20, random_state=42)

#Model
clf = Pipeline([
    ('vectorizer', CountVectorizer(analyzer="word",
                                   tokenizer=word_tokenize,
                                   max_features=None)),
    ('classifier', LinearSVC())
])

clf.fit(X_train, y_train)


#Measurements
pred_y = clf.predict(X_test)
print("Accuracy : ", metrics.accuracy_score(y_test, pred_y))

titles_options = [
    ("Confusion matrix, without normalization", None),
    ("Normalized confusion matrix", "true"),
]

for title, normalize in titles_options:
    disp = ConfusionMatrixDisplay.from_estimator(
        clf,
        X_test,
        y_test,
        display_labels=labels,
        cmap=plt.cm.Blues,
        normalize=normalize,
    )
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()
