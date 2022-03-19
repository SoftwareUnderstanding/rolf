from cgi import print_environ
from time import time
import pandas as pd
from sklearn import datasets
from Preprocessor import Preprocessor
from sklearn.preprocessing import LabelEncoder
import numpy as np
import csv
from sklearn.linear_model import LogisticRegression
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from ResultStorage import ResultStorage
from tqdm import tqdm
import fasttext
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from Vectorizing.CountVectorizer import getCountVectorizer
from Vectorizing.TF_IDF_Vectorizer import getWordLevelVectorizer, getNGramLevelVectorizer
from Embedding.WordEmbedding import createWordEmbedding
from Report import Report
from Report.CrossValidateNN import cross_validate_NN
from lstmModel import create_lstm_model, create_bidirec_lstm_model
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
import logthis

TEXT = "Text"
LABEL = "Label"
CV_splits = 5

def filter_dataframe(df, cat):
	#count = 0
	for ind, row in df.iterrows():
		if cat != str(row[LABEL]):
			#count += 1
			row[LABEL] = 'Other'	
	#print(f'{cat} filtered {count} rows in training dataset')

def get_sampling_strategy(df_train, categories: list, cat: str):
	sizes = df_train.groupby(LABEL).size()
	indexes = list(sizes.index)
	cat_size = sizes[indexes.index(cat)]

	# If the category is bigger than the sum of the other categories
	other_cat_size = int(cat_size/(len(df_train[LABEL].unique())-2))+1
	if cat_size > df_train.groupby(LABEL).size().sum() - cat_size:
		cat_size = df_train.groupby(LABEL).size().sum() - cat_size
	sampling_stratgy = {}
	change = 0
	for c in df_train[LABEL].unique():
		if c == cat:
			sampling_stratgy[c] = cat_size
		elif c not in categories:
			sampling_stratgy[c] = 0
		else:
			if sizes[indexes.index(c)] < other_cat_size:
				change = other_cat_size - sizes[indexes.index(c)]
			else: 
				change = 0
			sampling_stratgy[c] = min(other_cat_size+change, sizes[indexes.index(c)])
	logthis.say(f'Sampling strategy: {str(sampling_stratgy)}',)
	return sampling_stratgy



def train_models(train: str, test: str, categories = ["Natural Language Processing", "Computer Vision", "Sequential", "Audio", "Graphs", "Reinforcement Learning"]):
	logthis.say(f'Read files\nTrain dataset: {train} \nTest dataset: {test}')
	df_train = pd.read_csv(train, sep=';')
	df_test = pd.read_csv(test, sep = ';')
	df_train.drop_duplicates(subset=['Text'], inplace=True, keep=False)
	df_train = df_train.drop(columns = 'Repo')
	#df_test.drop( df_test[ df_test[LABEL] not in categories ].index , inplace=True)
	logthis.say('Read done')
	for i, cat in enumerate(categories):
		ind = i + 1
		logthis.say(f'Train test split starts for {cat=} category {ind}/{len(categories)}')
		x_train = df_train[TEXT].astype('U')
		y_train = df_train[LABEL]
		x_test = df_test[TEXT].astype('U')
		y_test = df_test[LABEL]

		undersample = RandomUnderSampler(sampling_strategy=get_sampling_strategy(df_train, categories, cat))
		#undersample = RandomUnderSampler(sampling_strategy='majority')
		x_train, y_train = undersample.fit_resample(x_train.to_frame(TEXT), y_train)

		logthis.say(f'Filter starts for {cat=} category {ind}/{len(categories)}')
		y_train = y_train.to_frame(LABEL)
		filter_dataframe(y_train, cat)
		y_test = y_test.to_frame(LABEL)
		filter_dataframe(y_test, cat)
		y_train = np.ravel(y_train)
		y_test = np.ravel(y_test)
		logthis.say(f'Filtering done')

		#countvect = CountVectorizer(analyzer="word",token_pattern=r'\w{1,}', max_features=10000, lowercase=True)
		#tfidf = TfidfVectorizer(analyzer="word",token_pattern=r'\w{1,}', max_features=10000, lowercase=True)

		result_storage = ResultStorage(train, test, cat)
	
		logthis.say(f'Logistic regression starts for {cat=} category {ind}/{len(categories)}')
		pipeline = Pipeline([
			('countvect', CountVectorizer(analyzer="word",token_pattern=r'\w{1,}', max_features=1500, lowercase=True, min_df=0.3)),
            ('lr', LogisticRegression(max_iter=10000, random_state=1))])
		result_storage.processResult(*Report.report(pipeline, train, test, x_train[TEXT], y_train, x_test, y_test, cat, name='LR_Count_Vectors_RandomUnder', cv=CV_splits, dict_scoring=Report.score_metrics, save=False))
		pipeline = Pipeline([
			('tfidf', TfidfVectorizer(analyzer="word",token_pattern=r'\w{1,}', max_features=1500, lowercase=True, min_df=0.3)),
            ('lr', LogisticRegression(max_iter=10000, random_state=1))])
		result_storage.processResult(*Report.report(pipeline, train, test, x_train[TEXT], y_train, x_test, y_test, cat, name='LR_TFIDF_RandomUnder', cv=CV_splits, dict_scoring=Report.score_metrics, save=False))
		
		logthis.say(f'SVC starts for {cat=} category {ind}/{len(categories)}')
		pipeline = Pipeline([
			('countvect', CountVectorizer(analyzer="word",token_pattern=r'\w{1,}', max_features=1500, lowercase=True, min_df=0.3)),
			('scaler', StandardScaler(with_mean=False)),
            ('lr', SVC(probability=True))])
		result_storage.processResult(*Report.report(pipeline, train, test, x_train[TEXT], y_train, x_test, y_test, cat, name='SVC_Count_Vectors_RandomUnder', cv=CV_splits, dict_scoring=Report.score_metrics, save=False))		
		pipeline = Pipeline([
			('tfidf', TfidfVectorizer(analyzer="word",token_pattern=r'\w{1,}', max_features=1500, lowercase=True, min_df=0.3)),
			('scaler', StandardScaler(with_mean=False)),
            ('lr', SVC(probability=True))])
		result_storage.processResult(*Report.report(pipeline, train, test, x_train[TEXT], y_train, x_test, y_test, cat, name='SVC_TFIDF_RandomUnder', cv=CV_splits, dict_scoring=Report.score_metrics, save=False))
		
		logthis.say(f'KNeighborsClassifier starts for {cat=} category {ind}/{len(categories)}')
		pipeline = Pipeline([
			('countvect', CountVectorizer(analyzer="word",token_pattern=r'\w{1,}', max_features=1500, lowercase=True, min_df=0.3)),
            ('lr', KNeighborsClassifier(n_neighbors=20, weights='distance', n_jobs=-1))])
		result_storage.processResult(*Report.report(pipeline, train, test, x_train[TEXT], y_train, x_test, y_test, cat, name='KNN_Count_Vectors_RandomUnder', cv=CV_splits, dict_scoring=Report.score_metrics, save=False))
		pipeline = Pipeline([
			('tfidf', TfidfVectorizer(analyzer="word",token_pattern=r'\w{1,}', max_features=1500, lowercase=True, min_df=0.3)),
            ('lr', KNeighborsClassifier(n_neighbors=20, weights='distance', n_jobs=-1))])
		result_storage.processResult(*Report.report(pipeline, train, test, x_train[TEXT], y_train, x_test, y_test, cat, name='KNN_TFIDF_RandomUnder', cv=CV_splits, dict_scoring=Report.score_metrics, save=False))
		
		logthis.say(f'RandomForestClassifier starts for {cat=} category {ind}/{len(categories)}')
		pipeline = Pipeline([
			('countvect', CountVectorizer(analyzer="word",token_pattern=r'\w{1,}', max_features=1500, lowercase=True, min_df=0.3)),
            ('lr', RandomForestClassifier(bootstrap=True,min_impurity_decrease=1e-7,n_jobs=-1, random_state=42))])
		result_storage.processResult(*Report.report(pipeline, train, test, x_train[TEXT], y_train, x_test, y_test, cat, name='RandomForestClassifier_Count_Vectors_RandomUnder', cv=CV_splits, dict_scoring=Report.score_metrics, save=False))
		pipeline = Pipeline([
			('tfidf', TfidfVectorizer(analyzer="word",token_pattern=r'\w{1,}', max_features=1500, lowercase=True, min_df=0.3)),
            ('lr', RandomForestClassifier(bootstrap=True,min_impurity_decrease=1e-7,n_jobs=-1, random_state=42))])
		result_storage.processResult(*Report.report(pipeline, train, test, x_train[TEXT], y_train, x_test, y_test, cat, name='RandomForestClassifier_TFIDF_RandomUnder', cv=CV_splits, dict_scoring=Report.score_metrics, save=False))
		

		logthis.say(f'LinearSVC starts for {cat=} category {ind}/{len(categories)}')
		pipeline = Pipeline([
			('countvect', CountVectorizer(analyzer="word",token_pattern=r'\w{1,}', max_features=1500, lowercase=True)),
            ('lr', CalibratedClassifierCV(LinearSVC()))])
		result_storage.processResult(*Report.report(pipeline, train, test, x_train[TEXT], y_train, x_test, y_test, cat, name='Linear_SVC_Count_Vectors_RandomUnder', cv=CV_splits, dict_scoring=Report.score_metrics, save=False))
		pipeline = Pipeline([
			('tfidf', TfidfVectorizer(analyzer="word",token_pattern=r'\w{1,}', max_features=1500, lowercase=True)),
            ('lr', CalibratedClassifierCV(LinearSVC()))])
		result_storage.processResult(*Report.report(pipeline, train, test, x_train[TEXT], y_train, x_test, y_test, cat, name='Linear_SVC_TFIDF_RandomUnder', cv=CV_splits, dict_scoring=Report.score_metrics, save=False))
	

		result_storage.dumpBestModel()
		result_storage.dumpResults()
