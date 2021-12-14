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
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import confusion_matrix
import seaborn as sns
import joblib
import random

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
    #print(X.head())
    #print(y.head())
    return X, y

df_train = pd.read_csv('dataset/train.csv', sep=';')
df_train_x = df_train['Text']
df_train_y = df_train['Label']


ngram_counter = CountVectorizer(analyzer="word",
                                   tokenizer=word_tokenize,
                                   max_features=None,
                                   lowercase=False)
df_train_x = ngram_counter.fit_transform(df_train_x)
#df_train_x, df_train_y = undersample(df_train_x.to_frame(), df_train_y.to_frame())

X_train, X_test, y_train, y_test = train_test_split(df_train_x, df_train_y, test_size=0.10, random_state=42, stratify=df_train_y)
#y_train = y_train.values.ravel()
#
# X_train = X_train.values.ravel()

#X_test  = ngram_counter.transform(X_test)
#print(X_test.info())

#X_train, X_test, y_train, y_test = train_test_split(df_train_x, df_train_y, test_size=0.10, random_state=42)

filename = 'model_balanced_undersample2.sav'
clf = joblib.load(filename)
pred_y = clf.predict(X_test)
print("Prediction")
print(pred_y)

unique, counts = np.unique(pred_y, return_counts=True)
plt.bar(unique, counts, 1)
plt.title('Class Frequency')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.show()

#print("Accuracy : ", metrics.accuracy_score(y_test, pred_y))



#cm = confusion_matrix(y_test, pred_y)
#y_unique = y_test.unique()
#cm_df = pd.DataFrame(cm,
#                     index = [y_unique], 
#                     columns = [y_unique])#

#plt.figure(figsize=(5,4))
#sns.heatmap(cm_df, annot=True)
#plt.title('Confusion Matrix')
#plt.ylabel('Actual Values')
#plt.xlabel('Predicted Values')
#plt.show()



test = {'Text':[]}
#Computer vision
#text = """https://paperswithcode.com/method/mask-r-cnn Mask R-CNN Mask R-CNN **Mask R-CNN** extends [Faster R-CNN](http://paperswithcode.com/method/faster-r-cnn) to solve instance segmentation tasks. It achieves this by adding a branch for predicting an object mask in parallel with the existing branch for bounding box recognition. In principle  Mask R-CNN is an intuitive extension of Faster [R-CNN](https://paperswithcode.com/method/r-cnn)  but constructing the mask branch properly is critical for good results. Most importantly  Faster R-CNN was not designed for pixel-to-pixel alignment between network inputs and outputs. This is evident in how [RoIPool](http://paperswithcode.com/method/roi-pooling)  the *de facto* core operation for attending to instances  performs coarse spatial quantization for feature extraction. To fix the misalignment  Mask R-CNN utilises a simple  quantization-free layer  called [RoIAlign](http://paperswithcode.com/method/roi-align)  that faithfully preserves exact spatial locations. Secondly  Mask R-CNN *decouples* mask and class prediction: it predicts a binary mask for each class independently  without competition among classes  and relies on the network's RoI classification branch to predict the category. In contrast  an [FCN](http://paperswithcode.com/method/fcn) usually perform per-pixel multi-class categorization  which couples segmentation and classification. mask-r-cnn 2000 None None https://github.com/facebookresearch/detectron2/blob/601d7666faaf7eb0ba64c9f9ce5811b13861fe12/detectron2/modeling/roi_heads/mask_head.py#L154 """
#NLP
#text = """https://paperswithcode.com/method/transformer Transformer Transformer A **Transformer** is a model architecture that eschews recurrence and instead relies entirely on an [attention mechanism](https://paperswithcode.com/methods/category/attention-mechanisms-1) to draw global dependencies between input and output. Before Transformers  the dominant sequence transduction models were based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The Transformer also employs an encoder and decoder  but removing recurrence in favor of [attention mechanisms](https://paperswithcode.com/methods/category/attention-mechanisms-1) allows for significantly more parallelization than methods like [RNNs](https://paperswithcode.com/methods/category/recurrent-neural-networks) and [CNNs](https://paperswithcode.com/methods/category/convolutional-neural-networks). attention-is-all-you-need 2000 None None https://github.com/tunz/transformer-pytorch/blob/e7266679f0b32fd99135ea617213f986ceede056/model/transformer.py#L201 """
text = """https://paperswithcode.com/method/skip-gram-word2vec Skip-gram Word2Vec Skip-gram Word2Vec **Skip-gram Word2Vec** is an architecture for computing word embeddings. Instead of using surrounding words to predict the center word  as with CBow Word2Vec  Skip-gram Word2Vec uses the central word to predict the surrounding words. The skip-gram objective function sums the log probabilities of the surrounding $n$ words to the left and right of the target word $w\_{t}$ to produce the following objective: $$J\_\theta = \frac{1}{T}\sum^{T}\_{t=1}\sum\_{-n\leq{j}\leq{n}  \neq{0}}\log{p}\left(w\_{j+1}\mid{w\_{t}}\right)$$ efficient-estimation-of-word-representations 2000 None None None """
text = text.replace('\n', ' ')
text = text.replace (',', ' ')
text = text.lower()
test['Text'].append(text)
X_test = pd.DataFrame(test, index=range(1))
X_test  = ngram_counter.transform(X_test)
#print(X_test)

pred_y = clf.predict(X_test)


for i in range(X_test.size):
    #print(X_test)
    #print(pred_y[i])
    #print(y_test[i])
    #t = X_test.iloc[i]
    yp = pred_y[i]
    #ya = y_test.iloc[i]
    print('Res:')
    #print(t +" ")
    print(yp)



#titles_options = [
#    ("Confusion matrix, without normalization", None),
#    ("Normalized confusion matrix", "true"),
#]

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

#plt.show()


