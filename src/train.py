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
	df =df_train.drop(df_train[df_train[LABEL] == 'General'].index)
	sizes = df.groupby(LABEL).size()
	indexes = list(sizes.index)
	cat_size = sizes[indexes.index(cat)]
	# If the category is bigger than the sum of the other categories
	#print(df[df[LABEL] != 'Other'].groupby(LABEL).size().sum())
	if cat_size > df.groupby(LABEL).size().sum() - cat_size:
		cat_size = df.groupby(LABEL).size().sum() - cat_size
	other_cat_size = int(cat_size/(len(categories)+1))+1
	sampling_strategy = {}
	change = 0
	for c in categories+['Other', 'General']:
		if c == cat:
			sampling_strategy[c] = cat_size
		elif c not in categories+['Other']:
			sampling_strategy[c] = 0
		else:
			s = other_cat_size+change
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

def train_models(train: str, out_folder: str, results_file:str, categories: List[str] = None, evaluation_metric: str = "test_f1-score_mean") -> None:
	if categories is None:
		categories = ["Sequential", "Audio", "Computer Vision","Graphs", "Reinforcement Learning", "Natural Language Processing", "Astrophisics"]
	print(categories)
	logthis.say(f'Read files\nTrain dataset: {train}')
	df_train = pd.read_csv(train, sep=';')
	#df_test.drop_duplicates(subset=['Text'], inplace=True, keep='last')
	df_train = df_train.drop(columns = 'Repo')
	logthis.say('Read done')
	for i, cat in enumerate(categories):
		ind = i + 1
		logthis.say(f'Train test split starts for {cat=} category {ind}/{len(categories)}')
		df_train = pd.read_csv(train, sep=';')
		#df_test.drop_duplicates(subset=['Text'], inplace=True, keep='last')
		df_train = df_train.drop(columns = 'Repo')
		x_train = df_train[TEXT].astype('U')
		y_train = df_train[LABEL]

		#undersample = RandomUnderSampler(sampling_strategy=get_sampling_strategy(df_train, categories, cat))
		#undersample = RandomUnderSampler(sampling_strategy='majority')
		#x_train, y_train = undersample.fit_resample(x_train.to_frame(TEXT), y_train)

		logthis.say(f'Filter starts for {cat=} category {ind}/{len(categories)}')
		y_train = y_train.to_frame(LABEL)
		print(y_train['Label'].unique())
		filter_dataframe(y_train, cat)
		print(y_train['Label'].unique())
		y_train = np.ravel(y_train)
		logthis.say(f'Filtering done')
		#logthis.say('Other: ' + str(np.count_nonzero(y_train == 'Other')))
		#logthis.say('CV: ' + str(np.count_nonzero(y_train == 'Computer Vision')))
		undersample = RandomUnderSampler(sampling_strategy='majority')
		x_train, y_train = undersample.fit_resample(x_train.to_frame(TEXT), y_train)
		#countvect = CountVectorizer(analyzer="word",token_pattern=r'\w{1,}', max_features=10000, lowercase=True)
		#tfidf = TfidfVectorizer(analyzer="word",token_pattern=r'\w{1,}', max_features=10000, lowercase=True)
		result_storage = ResultStorage(cat, evaluation_metric)

		logthis.say(f'Logistic regression starts for {cat=} category {ind}/{len(categories)}')
		pipeline = Pipeline([
			('countvect', CountVectorizer(max_df=0.9, min_df=0.1, max_features=None, ngram_range=(1, 1))),
            ('lr', LogisticRegression(max_iter=100000, C=10, penalty='l2', random_state=1))])
		result_storage.processResult(*Report.report(pipeline, train, x_train[TEXT], y_train, cat, name='LR_Count_Vectors_RandomUnder', cv=CV_splits, dict_scoring=Report.score_metrics, save=False))
		#logthis.say('###################################################')
		#logthis.say(pipeline.get_params())
		#logthis.say('###################################################')
		pipeline = Pipeline([
			('tfidf', TfidfVectorizer(max_df=0.9, min_df=0.1, max_features=None, ngram_range=(1, 1))),
            ('lr', LogisticRegression(max_iter=10000, C=10, penalty='l2', random_state=1))])
		result_storage.processResult(*Report.report(pipeline, train, x_train[TEXT], y_train, cat, name='LR_TFIDF_RandomUnder', cv=CV_splits, dict_scoring=Report.score_metrics, save=False))
		#logthis.say('###################################################')
		#logthis.say(pipeline.get_params())
		#logthis.say('###################################################')

		logthis.say(f'SVC starts for {cat=} category {ind}/{len(categories)}')
		pipeline = Pipeline([
			('countvect', CountVectorizer(max_df=0.9, min_df=0.1, max_features=None, ngram_range=(1, 1))),
            ('svc', SVC(probability=True, C=100, gamma=0.1, kernel='rbf'))])
		result_storage.processResult(*Report.report(pipeline, train, x_train[TEXT], y_train, cat, name='SVC_Count_Vectors_RandomUnder', cv=CV_splits, dict_scoring=Report.score_metrics, save=False))		
		#logthis.say('###################################################')
		#logthis.say(pipeline.get_params())
		#logthis.say('###################################################')
		pipeline = Pipeline([
			('tfidf', TfidfVectorizer(max_df=0.9, min_df=0.1, max_features=None, ngram_range=(1, 1))),
            ('svc', SVC(probability=True, C=100, gamma=0.1, kernel='rbf'))])
		result_storage.processResult(*Report.report(pipeline, train, x_train[TEXT], y_train, cat, name='SVC_TFIDF_RandomUnder', cv=CV_splits, dict_scoring=Report.score_metrics, save=False))
		#logthis.say('###################################################')
		#logthis.say(pipeline.get_params())
		#logthis.say('###################################################')

		logthis.say(f'KNeighborsClassifier starts for {cat=} category {ind}/{len(categories)}')
		pipeline = Pipeline([
			('countvect', CountVectorizer(max_df=0.9, min_df=0.1, max_features=None, ngram_range=(1, 1))),
            ('knn', KNeighborsClassifier(leaf_size=40, metric='minkowski', n_neighbors=20, p=2, weights='distance'))])
		result_storage.processResult(*Report.report(pipeline, train, x_train[TEXT], y_train, cat, name='KNN_Count_Vectors_RandomUnder', cv=CV_splits, dict_scoring=Report.score_metrics, save=False))
		#logthis.say('###################################################')
		#logthis.say(pipeline.get_params())
		#logthis.say('###################################################')
		pipeline = Pipeline([
			('tfidf', TfidfVectorizer(max_df=0.9, min_df=0.1, max_features=None, ngram_range=(1, 1))),
            ('knn', KNeighborsClassifier(leaf_size=40, metric='minkowski', n_neighbors=20, p=2, weights='distance'))])
		result_storage.processResult(*Report.report(pipeline, train, x_train[TEXT], y_train, cat, name='KNN_TFIDF_RandomUnder', cv=CV_splits, dict_scoring=Report.score_metrics, save=False))
		#logthis.say('###################################################')
		#logthis.say(pipeline.get_params())
		#logthis.say('###################################################')

		logthis.say(f'RandomForestClassifier starts for {cat=} category {ind}/{len(categories)}')
		pipeline = Pipeline([
			('countvect', CountVectorizer(max_df=0.9, min_df=0.1, max_features=None, ngram_range=(1, 1))),
            ('random_forest', RandomForestClassifier(max_features='sqrt', random_state=42, bootstrap=False, max_depth=None, min_samples_leaf=1, min_samples_split=5, n_estimators=100))])
		result_storage.processResult(*Report.report(pipeline, train, x_train[TEXT], y_train, cat, name='RandomForestClassifier_Count_Vectors_RandomUnder', cv=CV_splits, dict_scoring=Report.score_metrics, save=False))
		#logthis.say('###################################################')
		#logthis.say(pipeline.get_params())
		#logthis.say('###################################################')
		pipeline = Pipeline([
			('tfidf', TfidfVectorizer(max_df=0.9, min_df=0.1, max_features=None, ngram_range=(1, 1))),
            ('random_forest', RandomForestClassifier(max_features='sqrt', random_state=42, bootstrap=False, max_depth=None, min_samples_leaf=1, min_samples_split=5, n_estimators=100))])
		result_storage.processResult(*Report.report(pipeline, train, x_train[TEXT], y_train, cat, name='RandomForestClassifier_TFIDF_RandomUnder', cv=CV_splits, dict_scoring=Report.score_metrics, save=False))
		#logthis.say('###################################################')
		#logthis.say(pipeline.get_params())
		#logthis.say('###################################################')

		logthis.say(f'LinearSVC starts for {cat=} category {ind}/{len(categories)}')
		pipeline = Pipeline([
			('countvect', CountVectorizer(max_df=0.9, min_df=0.1, max_features=None, ngram_range=(1, 1))),
            ('linear_svc', LinearSVC(C=0.0001, dual=False))])
		result_storage.processResult(*Report.report(pipeline, train, x_train[TEXT], y_train, cat, name='Linear_SVC_Count_Vectors_RandomUnder', cv=CV_splits, dict_scoring=Report.score_metrics, save=False))
		#logthis.say('###################################################')
		#logthis.say(pipeline.get_params())
		#logthis.say('###################################################')
		pipeline = Pipeline([
			('tfidf', TfidfVectorizer(max_df=0.9, min_df=0.1, max_features=None, ngram_range=(1, 1))),
            ('linear_svc', LinearSVC(C=0.0001, dual=False))])
		result_storage.processResult(*Report.report(pipeline, train, x_train[TEXT], y_train, cat, name='Linear_SVC_TFIDF_RandomUnder', cv=CV_splits, dict_scoring=Report.score_metrics, save=False))
		#logthis.say('###################################################')
		#logthis.say(pipeline.get_params())
		#logthis.say('###################################################')

		result_storage.dumpBestModel(out_folder)
		result_storage.dumpResults(results_file)
