#from cgi import print_environ
#from time import time
from pathlib import Path
from typing import Dict, List
import pandas as pd
#from sklearn import datasets
#from Preprocessor import Preprocessor
#from sklearn.preprocessing import LabelEncoder
import numpy as np
import csv
from sklearn.linear_model import LogisticRegression
from imblearn.under_sampling import RandomUnderSampler
#from sklearn.utils.class_weight import compute_class_weight
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
#from sklearn.linear_model import SGDClassifier
#from sklearn.ensemble import AdaBoostClassifier
#from sklearn.naive_bayes import MultinomialNB
from ResultStorage import ResultStorage
#from tqdm import tqdm
#import fasttext
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

#from Vectorizing.CountVectorizer import getCountVectorizer
#from Vectorizing.TF_IDF_Vectorizer import getWordLevelVectorizer, getNGramLevelVectorizer
#from Embedding.WordEmbedding import createWordEmbedding
from Report import Report
#from Report.CrossValidateNN import cross_validate_NN
#from lstmModel import create_lstm_model, create_bidirec_lstm_model
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
import logthis

TEXT = "Text"
LABEL = "Label"
CV_splits = 5

def filter_dataframe(df: pd.DataFrame, category: str) -> None:
	for ind, row in df.iterrows():
		if category != str(row[LABEL]):
			row[LABEL] = 'Other'	
	#print(f'{cat} filtered {count} rows in training dataset')

def get_sampling_strategy(df_train: pd.DataFrame, categories: List[str], cat: str) -> Dict[str, int]:
	sizes = df_train.groupby(LABEL).size()
	indexes = list(sizes.index)
	cat_size = sizes[indexes.index(cat)]

	# If the category is bigger than the sum of the other categories
	other_cat_size = int(cat_size/(len(df_train[LABEL].unique())-2))+1
	if cat_size > df_train.groupby(LABEL).size().sum() - cat_size:
		cat_size = df_train.groupby(LABEL).size().sum() - cat_size
	sampling_strategy = {}
	change = 0
	for c in df_train[LABEL].unique():
		if c == cat:
			sampling_strategy[c] = cat_size
		elif c not in categories:
			sampling_strategy[c] = 0
		else:
			if sizes[indexes.index(c)] < other_cat_size:
				change = other_cat_size - sizes[indexes.index(c)]
			else: 
				change = 0
			sampling_strategy[c] = min(other_cat_size+change, sizes[indexes.index(c)])
	logthis.say(f'Sampling strategy: {str(sampling_strategy)}',)
	return sampling_strategy

def train_models(train: str, out_folder: str, results_file:str, categories: List[str] = None, evaluation_metric: str = "test_f1-score_mean") -> None:
	if categories is None:
		categories = ["Natural Language Processing", "Computer Vision", "Sequential", "Audio", "Graphs", "Reinforcement Learning"]
	logthis.say(f'Read files\nTrain dataset: {train}')
	df_train = pd.read_csv(train, sep=';')
	#df_test.drop_duplicates(subset=['Text'], inplace=True, keep='last')
	df_train = df_train.drop(columns = 'Repo')
	logthis.say('Read done')
	for i, cat in enumerate(categories):
		ind = i + 1
		logthis.say(f'Train test split starts for {cat=} category {ind}/{len(categories)}')
		x_train = df_train[TEXT].astype('U')
		y_train = df_train[LABEL]

		undersample = RandomUnderSampler(sampling_strategy=get_sampling_strategy(df_train, categories, cat))
		#undersample = RandomUnderSampler(sampling_strategy='majority')
		x_train, y_train = undersample.fit_resample(x_train.to_frame(TEXT), y_train)

		logthis.say(f'Filter starts for {cat=} category {ind}/{len(categories)}')
		y_train = y_train.to_frame(LABEL)
		filter_dataframe(y_train, cat)
		y_train = np.ravel(y_train)
		logthis.say(f'Filtering done')

		#undersample = RandomUnderSampler(sampling_strategy='majority')
		#x_train, y_train = undersample.fit_resample(x_train.to_frame(TEXT), y_train)
		#countvect = CountVectorizer(analyzer="word",token_pattern=r'\w{1,}', max_features=10000, lowercase=True)
		#tfidf = TfidfVectorizer(analyzer="word",token_pattern=r'\w{1,}', max_features=10000, lowercase=True)

		result_storage = ResultStorage(train, cat, evaluation_metric)

		logthis.say(f'Logistic regression starts for {cat=} category {ind}/{len(categories)}')
		pipeline = Pipeline([
			('countvect', CountVectorizer(max_df=1.0, min_df=0.0, max_features=None, ngram_range=(1, 2))),
            ('lr', LogisticRegression(max_iter=10000, random_state=1))])
		result_storage.processResult(*Report.report(pipeline, train, x_train[TEXT], y_train, cat, name='LR_Count_Vectors_RandomUnder', cv=CV_splits, dict_scoring=Report.score_metrics, save=False))
		pipeline = Pipeline([
			('tfidf', TfidfVectorizer(max_df=1.0, min_df=0.0, max_features=None, ngram_range=(1, 2))),
            ('lr', LogisticRegression(max_iter=10000, random_state=1))])
		result_storage.processResult(*Report.report(pipeline, train, x_train[TEXT], y_train, cat, name='LR_TFIDF_RandomUnder', cv=CV_splits, dict_scoring=Report.score_metrics, save=False))
		
		logthis.say(f'SVC starts for {cat=} category {ind}/{len(categories)}')
		pipeline = Pipeline([
			('countvect', CountVectorizer(max_df=1.0, min_df=0.0, max_features=None, ngram_range=(1, 2))),
            ('svc', SVC(probability=True))])
		result_storage.processResult(*Report.report(pipeline, train, x_train[TEXT], y_train, cat, name='SVC_Count_Vectors_RandomUnder', cv=CV_splits, dict_scoring=Report.score_metrics, save=False))		
		pipeline = Pipeline([
			('tfidf', TfidfVectorizer(max_df=1.0, min_df=0.0, max_features=None, ngram_range=(1, 2))),
            ('svc', SVC(probability=True))])
		result_storage.processResult(*Report.report(pipeline, train, x_train[TEXT], y_train, cat, name='SVC_TFIDF_RandomUnder', cv=CV_splits, dict_scoring=Report.score_metrics, save=False))
		
		logthis.say(f'KNeighborsClassifier starts for {cat=} category {ind}/{len(categories)}')
		pipeline = Pipeline([
			('countvect', CountVectorizer(max_df=1.0, min_df=0.0, max_features=None, ngram_range=(1, 2))),
            ('knn', KNeighborsClassifier())])
		result_storage.processResult(*Report.report(pipeline, train, x_train[TEXT], y_train, cat, name='KNN_Count_Vectors_RandomUnder', cv=CV_splits, dict_scoring=Report.score_metrics, save=False))
		pipeline = Pipeline([
			('tfidf', TfidfVectorizer(max_df=1.0, min_df=0.0, max_features=None, ngram_range=(1, 2))),
            ('knn', KNeighborsClassifier())])
		result_storage.processResult(*Report.report(pipeline, train, x_train[TEXT], y_train, cat, name='KNN_TFIDF_RandomUnder', cv=CV_splits, dict_scoring=Report.score_metrics, save=False))
		
		logthis.say(f'RandomForestClassifier starts for {cat=} category {ind}/{len(categories)}')
		pipeline = Pipeline([
			('countvect', CountVectorizer(max_df=1.0, min_df=0.0, max_features=None, ngram_range=(1, 2))),
            ('random_forest', RandomForestClassifier(random_state=42))])
		result_storage.processResult(*Report.report(pipeline, train, x_train[TEXT], y_train, cat, name='RandomForestClassifier_Count_Vectors_RandomUnder', cv=CV_splits, dict_scoring=Report.score_metrics, save=False))
		pipeline = Pipeline([
			('tfidf', TfidfVectorizer(max_df=1.0, min_df=0.0, max_features=None, ngram_range=(1, 2))),
            ('random_forest', RandomForestClassifier(random_state=42))])
		result_storage.processResult(*Report.report(pipeline, train, x_train[TEXT], y_train, cat, name='RandomForestClassifier_TFIDF_RandomUnder', cv=CV_splits, dict_scoring=Report.score_metrics, save=False))
		
		logthis.say(f'LinearSVC starts for {cat=} category {ind}/{len(categories)}')
		pipeline = Pipeline([
			('countvect', CountVectorizer(max_df=1.0, min_df=0.0, max_features=None, ngram_range=(1, 2))),
            ('linear_svc', LinearSVC())])
		result_storage.processResult(*Report.report(pipeline, train, x_train[TEXT], y_train, cat, name='Linear_SVC_Count_Vectors_RandomUnder', cv=CV_splits, dict_scoring=Report.score_metrics, save=False))
		pipeline = Pipeline([
			('tfidf', TfidfVectorizer(max_df=1.0, min_df=0.0, max_features=None, ngram_range=(1, 2))),
            ('linear_svc', LinearSVC())])
		result_storage.processResult(*Report.report(pipeline, train, x_train[TEXT], y_train, cat, name='Linear_SVC_TFIDF_RandomUnder', cv=CV_splits, dict_scoring=Report.score_metrics, save=False))

		result_storage.dumpBestModel(out_folder)
		result_storage.dumpResults(results_file)
