import os
import argparse
import sys
import json
import pandas as pd
import numpy as np
import nltk
import time
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

labels = ['Computer Vision', 'Natural Language Processing']

df_train = pd.read_csv('dataset/train_all.csv', sep=';')
df_train_x = df_train['Text']
df_train_y = df_train['Label']
print(df_train.shape)
print(df_train.head())



#Plot the label's distribution
#unique, counts = np.unique(df_train_y, return_counts=True)
#plt.bar(unique, counts, 1)
#plt.title('Class Frequency')
#plt.xlabel('Class')
#plt.ylabel('Frequency')
#plt.show()


data_somef = {'Text':[], 'Label':[]}

"""
for index, i in df_train.iterrows():
#for j in range(1):
    repo = i['Repo']
    label = i['Label']
    print(repo)
    os.system("somef describe -r {} -o tmp.json -t 0.8".format(repo))
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
X_train, X_test, y_train, y_test = train_test_split(df_train_x, df_train_y, test_size=0.10, random_state=42)

#Model
clf = Pipeline([
    ('vectorizer', CountVectorizer(analyzer="word",
                                   tokenizer=word_tokenize,
                                   max_features=None)),
    ('classifier', LinearSVC())
])

clf.fit(X_train, y_train)
pred_y = clf.predict(X_test)



#Measurements
from sklearn.metrics import confusion_matrix
import seaborn as sns


print("Accuracy : ", metrics.accuracy_score(y_test, pred_y))

#titles_options = [
#    ("Confusion matrix, without normalization", None),
#    ("Normalized confusion matrix", "true"),
#]

cm = confusion_matrix(y_test, pred_y)
y_unique = y_test.unique()
cm_df = pd.DataFrame(cm,
                     index = [y_unique], 
                     columns = [y_unique])

plt.figure(figsize=(5,4))
sns.heatmap(cm_df, annot=True)
plt.title('Confusion Matrix')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.show()


#for title, normalize in titles_options:
#    disp = ConfusionMatrixDisplay.from_estimator(
#        clf,
#        X_test,
#        y_test,
#        display_labels=labels,
#        cmap=plt.cm.Blues,
#        normalize=normalize,
#    )
#    disp.ax_.set_title(title)

#    print(title)
#    print(disp.confusion_matrix)

plt.show()
