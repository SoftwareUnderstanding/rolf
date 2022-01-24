import os
import argparse
import sys
import json
import pandas as pd
import numpy as np
import nltk
import time
from sklearn.metrics import confusion_matrix
from nltk import word_tokenize
from pandas.core.indexes.base import Index
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_validate, cross_val_score, cross_val_predict
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_fscore_support, make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import pickle
import random
import pkg_resources
#import somef
import seaborn as sns
#from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import csv
from datetime import datetime
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

import nltk
nltk.download('punkt')


class DataframeContainer:
    def __init__(self, name, inputFilename):
        self.name = name
        self.inputFilename = inputFilename
        self.dataframe = pd.read_csv(inputFilename, sep=';')
        
    def filter_dataframe(self):
        count = 0
        for ind, row in self.dataframe.iterrows():
            if self.name != str(row['Label']):
                count += 1
                row['Label'] = 'Other'
                row['Text'] = row['Text'].replace('\n', ' ').replace(',', ' ').lower()
        print(f'{self.name} filtered {count} rows')  

    def separate_x_y(self):
        self.df_X, self.df_y = self.dataframe['Text'], self.dataframe['Label']
        #unique, counts = np.unique(self.df_y , return_counts=True)
        #plt.bar(unique, counts, 1)
        #plt.title('Class Frequency')
        #plt.xlabel('Class')
        #plt.ylabel('Frequency')
        #plt.show()
    
    def split_train_test(self, test_size = 0.2, random_state = 42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.df_X, self.df_y, test_size=test_size, random_state=random_state, stratify=self.df_y)

    def clf_fit_cv_ru_lsvc(self):
        self.clf = Pipeline([
            ('vectorizer', CountVectorizer(analyzer="word",
                                           tokenizer=word_tokenize,
                                           max_features=None,
                                           lowercase=True)),
            ('undersample', RandomUnderSampler(sampling_strategy='majority')),
            ('linearsvc', LinearSVC(random_state=42))
        ])
        self.clf.fit(self.X_train, self.y_train)
        print(f'{self.name} clf fit done')
    
    def clf_fit_tfidf_ru_lsvc(self):
        self.clf = Pipeline([
            ('tfidf', TfidfVectorizer(analyzer="word",
                                           tokenizer=word_tokenize,
                                           max_features=None,
                                           lowercase=True)),
            ('undersample', RandomUnderSampler(sampling_strategy='majority')),
            ('linearsvc', LinearSVC(random_state=42))
        ])
        self.clf.fit(self.X_train, self.y_train)
        print(f'{self.name} clf fit done')
        
    def clf_fit_tfidf_ru_rfc(self):
        self.clf = Pipeline([
            ('tfidf', TfidfVectorizer(analyzer="word",
                                           tokenizer=word_tokenize,
                                           max_features=None,
                                           lowercase=True)),
            ('undersample', RandomUnderSampler(sampling_strategy='majority')),
            ('randomforest', RandomForestClassifier(max_depth=None,random_state=1))
        ])
        self.clf.fit(self.X_train, self.y_train)
        print(f'{self.name} clf fit done')
    
    def clf_fit_cv_ru_rfc(self):
        self.clf = Pipeline([
            ('vectorizer', CountVectorizer(analyzer="word",
                                           tokenizer=word_tokenize,
                                           max_features=None,
                                           lowercase=True)),
            ('undersample', RandomUnderSampler(sampling_strategy='majority')),
            ('randomforest', RandomForestClassifier(max_depth=None,random_state=1))
        ])
        self.clf.fit(self.X_train, self.y_train)
        print(f'{self.name} clf fit done')

    def clf_fit_tfidf_ru_mnb(self):
        self.clf = Pipeline([
            ('tfidf', TfidfVectorizer(analyzer="word",
                                           tokenizer=word_tokenize,
                                           max_features=None,
                                           lowercase=True)),
            ('undersample', RandomUnderSampler(sampling_strategy='majority')),
            ('multinomualnb', MultinomialNB())
        ])
        self.clf.fit(self.X_train, self.y_train)
        print(f'{self.name} clf fit done')
    
    def clf_fit_cv_ru_mnb(self):
        self.clf = Pipeline([
            ('vectorizer', CountVectorizer(analyzer="word",
                                           tokenizer=word_tokenize,
                                           max_features=None,
                                           lowercase=True)),
            ('undersample', RandomUnderSampler(sampling_strategy='majority')),
            ('multinomialnb', MultinomialNB())
        ])
        self.clf.fit(self.X_train, self.y_train)
        print(f'{self.name} clf fit done')

    def predict(self):
        self.y_pred = self.clf.predict(self.X_test)

    def confusion_matrix_macro(self):
        print("Metrics macro")
        y_unique = self.y_test.unique()
        cm = confusion_matrix(self.y_test, self.y_pred, labels=y_unique)
        cm_df = pd.DataFrame(cm, index = [y_unique], columns = [y_unique])
        #plt.figure(figsize=(5,4))
        #sns.heatmap(cm_df, annot=True, fmt='d')
        #plt.title('Confusion Matrix')
        #plt.ylabel('Actual Values')
        #plt.xlabel('Predicted Values')
        #plt.show()
        print(f"Accuracy {self.name} : {metrics.accuracy_score(self.y_test, self.y_pred)}")
        m = metrics.precision_recall_fscore_support(self.y_test, self.y_pred, average='macro')
        self.metrics_type = 'macro'
        self.precision = m[0]
        self.recall = m[1]
        self.f1score = m[2]
        print(f"Precision {self.name} : {m[0]} \nRecall {self.name} : {m[1]} \nF1-score {self.name} : {m[2]}")
    
    def confusion_matrix_weighted(self):
        print("Metrics weighted")
        y_unique = self.y_test.unique()
        cm = confusion_matrix(self.y_test, self.y_pred, labels=y_unique)
        cm_df = pd.DataFrame(cm, index = [y_unique], columns = [y_unique])
        #plt.figure(figsize=(5,4))
        #sns.heatmap(cm_df, annot=True, fmt='d')
        #plt.title('Confusion Matrix')
        #plt.ylabel('Actual Values')
        #plt.xlabel('Predicted Values')
        #plt.show()
        print(f"Accuracy {self.name} : {metrics.accuracy_score(self.y_test, self.y_pred)}")
        m = metrics.precision_recall_fscore_support(self.y_test, self.y_pred, average='weighted')
        self.metrics_type = 'weighted'
        self.precision = m[0]
        self.recall = m[1]
        self.f1score = m[2]
        print(f"Precision {self.name} : {m[0]} \nRecall {self.name} : {m[1]} \nF1-score {self.name} : {m[2]}")

    def cv(self):
        scores = cross_val_score(self.clf, self.df_X, self.df_y, cv=3, scoring='f1')
        print(scores)
    
    def save_pickle(self):
        self.currentDatetime = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        self.pickleFilename = f"{self.name}.sav"
        Path('../results/models/' + self.currentDatetime).mkdir(exist_ok=True)
        pickle.dump(self.clf, open('../results/models/' + self.currentDatetime + '/' + self.pickleFilename, 'wb'))

    def printScoreboard(self):
        csvFileName = f"{self.name.lower().replace(' ', '_')}.csv"
        csvExists = os.path.exists('../results/scoreboards/' + csvFileName)
        with open('../results/scoreboards/' + csvFileName, 'a+') as csvfile:
            csvWriter = csv.writer(csvfile, delimiter=';')
            if not csvExists:
                csvWriter.writerow(["Pipeline", "Input", "F1 score", "Precision", "Recall", "Pickle file name", "Datetime", "Sklearn version"])
            csvWriter.writerow([self.clf.steps, 
                                self.inputFilename, 
                                str(self.f1score) + ' ' +  self.metrics_type, 
                                str(self.precision) + ' ' +  str(self.metrics_type), 
                                str(self.recall) + ' ' +  self.metrics_type, 
                                self.currentDatetime + '/' + self.pickleFilename, 
                                self.currentDatetime, 
                                '1.0'])


names_list = ["Audio", "Computer Vision", "General", "Graphs", "Natural Language Processing", "Reinforcement Learning", "Sequential"]

### Somef dataset -----------------------------------------------------------------------------------------------------------------------

dataframecontainers_list = [DataframeContainer(name, '../data/somef_data.csv') for name in names_list]


## CountVectorizer + RandomUndersampling + LineraSVC -----------------------------------------------------------------------------------------------------------------------

for container in dataframecontainers_list:
    container.filter_dataframe()
    container.separate_x_y()
    container.split_train_test()
    container.clf_fit_cv_ru_lsvc()
    container.predict()
    container.save_pickle()
    container.confusion_matrix_macro()
    container.printScoreboard()
    container.confusion_matrix_weighted()
    container.printScoreboard()

## TF+IDF + RandomUndersampling + LineraSVC -----------------------------------------------------------------------------------------------------------------------

for container in dataframecontainers_list:
    container.filter_dataframe()
    container.separate_x_y()
    container.split_train_test()
    container.clf_fit_tfidf_ru_lsvc()
    container.predict()
    container.save_pickle()
    container.confusion_matrix_macro()
    container.printScoreboard()
    container.confusion_matrix_weighted()
    container.printScoreboard()

## TF+IDF + RandomUndersampling + RandomForestClassifier -----------------------------------------------------------------------------------------------------------------------

for container in dataframecontainers_list:
    container.filter_dataframe()
    container.separate_x_y()
    container.split_train_test()
    container.clf_fit_tfidf_ru_rfc()
    container.predict()
    container.save_pickle()
    container.confusion_matrix_macro()
    container.printScoreboard()
    container.confusion_matrix_weighted()
    container.printScoreboard()

## CountVectorizer + RandomUndersampling + RandomForestClassifier -----------------------------------------------------------------------------------------------------------------------

for container in dataframecontainers_list:
    container.filter_dataframe()
    container.separate_x_y()
    container.split_train_test()
    container.clf_fit_cv_ru_rfc()
    container.predict()
    container.save_pickle()
    container.confusion_matrix_macro()
    container.printScoreboard()
    container.confusion_matrix_weighted()
    container.printScoreboard()

## TF+IDF + RandomUndersampling + MultinomialNB -----------------------------------------------------------------------------------------------------------------------

for container in dataframecontainers_list:
    container.filter_dataframe()
    container.separate_x_y()
    container.split_train_test()
    container.clf_fit_tfidf_ru_mnb()
    container.predict()
    container.save_pickle()
    container.confusion_matrix_macro()
    container.printScoreboard()
    container.confusion_matrix_weighted()
    container.printScoreboard()


## CountVectorizer + RandomUndersampling + MultinomialNB -----------------------------------------------------------------------------------------------------------------------

for container in dataframecontainers_list:
    container.filter_dataframe()
    container.separate_x_y()
    container.split_train_test()
    container.clf_fit_cv_ru_mnb()
    container.predict()
    container.save_pickle()
    container.confusion_matrix_macro()
    container.printScoreboard()
    container.confusion_matrix_weighted()
    container.printScoreboard()








### PapersWithCode dataset -----------------------------------------------------------------------------------------------------------------------

dataframecontainers_list = [DataframeContainer(name, '../data/train_all.csv') for name in names_list]


## CountVectorizer + RandomUndersampling + LineraSVC -----------------------------------------------------------------------------------------------------------------------

for container in dataframecontainers_list:
    container.filter_dataframe()
    container.separate_x_y()
    container.split_train_test()
    container.clf_fit_cv_ru_lsvc()
    container.predict()
    container.save_pickle()
    container.confusion_matrix_macro()
    container.printScoreboard()
    container.confusion_matrix_weighted()
    container.printScoreboard()

## TF+IDF + RandomUndersampling + LineraSVC -----------------------------------------------------------------------------------------------------------------------

for container in dataframecontainers_list:
    container.filter_dataframe()
    container.separate_x_y()
    container.split_train_test()
    container.clf_fit_tfidf_ru_lsvc()
    container.predict()
    container.save_pickle()
    container.confusion_matrix_macro()
    container.printScoreboard()
    container.confusion_matrix_weighted()
    container.printScoreboard()

## TF+IDF + RandomUndersampling + RandomForestClassifier -----------------------------------------------------------------------------------------------------------------------

for container in dataframecontainers_list:
    container.filter_dataframe()
    container.separate_x_y()
    container.split_train_test()
    container.clf_fit_tfidf_ru_rfc()
    container.predict()
    container.save_pickle()
    container.confusion_matrix_macro()
    container.printScoreboard()
    container.confusion_matrix_weighted()
    container.printScoreboard()

## CountVectorizer + RandomUndersampling + RandomForestClassifier -----------------------------------------------------------------------------------------------------------------------

for container in dataframecontainers_list:
    container.filter_dataframe()
    container.separate_x_y()
    container.split_train_test()
    container.clf_fit_cv_ru_rfc()
    container.predict()
    container.save_pickle()
    container.confusion_matrix_macro()
    container.printScoreboard()
    container.confusion_matrix_weighted()
    container.printScoreboard()

## TF+IDF + RandomUndersampling + MultinomialNB -----------------------------------------------------------------------------------------------------------------------

for container in dataframecontainers_list:
    container.filter_dataframe()
    container.separate_x_y()
    container.split_train_test()
    container.clf_fit_tfidf_ru_mnb()
    container.predict()
    container.save_pickle()
    container.confusion_matrix_macro()
    container.printScoreboard()
    container.confusion_matrix_weighted()
    container.printScoreboard()


## CountVectorizer + RandomUndersampling + MultinomialNB -----------------------------------------------------------------------------------------------------------------------

for container in dataframecontainers_list:
    container.filter_dataframe()
    container.separate_x_y()
    container.split_train_test()
    container.clf_fit_cv_ru_mnb()
    container.predict()
    container.save_pickle()
    container.confusion_matrix_macro()
    container.printScoreboard()
    container.confusion_matrix_weighted()
    container.printScoreboard()











### Merged dataset -----------------------------------------------------------------------------------------------------------------------

dataframecontainers_list = [DataframeContainer(name, '../data/merged.csv') for name in names_list]


## CountVectorizer + RandomUndersampling + LineraSVC -----------------------------------------------------------------------------------------------------------------------

for container in dataframecontainers_list:
    container.filter_dataframe()
    container.separate_x_y()
    container.split_train_test()
    container.clf_fit_cv_ru_lsvc()
    container.predict()
    container.save_pickle()
    container.confusion_matrix_macro()
    container.printScoreboard()
    container.confusion_matrix_weighted()
    container.printScoreboard()

## TF+IDF + RandomUndersampling + LineraSVC -----------------------------------------------------------------------------------------------------------------------

for container in dataframecontainers_list:
    container.filter_dataframe()
    container.separate_x_y()
    container.split_train_test()
    container.clf_fit_tfidf_ru_lsvc()
    container.predict()
    container.save_pickle()
    container.confusion_matrix_macro()
    container.printScoreboard()
    container.confusion_matrix_weighted()
    container.printScoreboard()

## TF+IDF + RandomUndersampling + RandomForestClassifier -----------------------------------------------------------------------------------------------------------------------

for container in dataframecontainers_list:
    container.filter_dataframe()
    container.separate_x_y()
    container.split_train_test()
    container.clf_fit_tfidf_ru_rfc()
    container.predict()
    container.save_pickle()
    container.confusion_matrix_macro()
    container.printScoreboard()
    container.confusion_matrix_weighted()
    container.printScoreboard()

## CountVectorizer + RandomUndersampling + RandomForestClassifier -----------------------------------------------------------------------------------------------------------------------

for container in dataframecontainers_list:
    container.filter_dataframe()
    container.separate_x_y()
    container.split_train_test()
    container.clf_fit_cv_ru_rfc()
    container.predict()
    container.save_pickle()
    container.confusion_matrix_macro()
    container.printScoreboard()
    container.confusion_matrix_weighted()
    container.printScoreboard()

## TF+IDF + RandomUndersampling + MultinomialNB -----------------------------------------------------------------------------------------------------------------------

for container in dataframecontainers_list:
    container.filter_dataframe()
    container.separate_x_y()
    container.split_train_test()
    container.clf_fit_tfidf_ru_mnb()
    container.predict()
    container.save_pickle()
    container.confusion_matrix_macro()
    container.printScoreboard()
    container.confusion_matrix_weighted()
    container.printScoreboard()


## CountVectorizer + RandomUndersampling + MultinomialNB -----------------------------------------------------------------------------------------------------------------------

for container in dataframecontainers_list:
    container.filter_dataframe()
    container.separate_x_y()
    container.split_train_test()
    container.clf_fit_cv_ru_mnb()
    container.predict()
    container.save_pickle()
    container.confusion_matrix_macro()
    container.printScoreboard()
    container.confusion_matrix_weighted()
    container.printScoreboard()
