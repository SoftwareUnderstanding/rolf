from cgi import print_environ
from time import time
from typing import List
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
import numpy as np
import csv
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.svm import SVC, LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from tqdm import tqdm
import fasttext
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV
import logthis

from src.ResultStorage import ResultStorage

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



def train_models(train: str, test: str, out_folder: str, results_file:str, categories: List[str] = None, evaluation_metric: str = "test_f1-score_mean") -> None:
	if categories is None:
		categories = ["Natural Language Processing", "Computer Vision", "Sequential", "Audio", "Graphs", "Reinforcement Learning"]
	logthis.say(f'Read files\nTrain dataset: {train} \nTest dataset: {test}')
	df_train = pd.read_csv(train, sep=';')
	df_test = pd.read_csv(test, sep = ';')
	#df_test.drop_duplicates(subset=['Text'], inplace=True, keep='last')
	df_train = df_train.drop(columns = 'Repo')
	logthis.say('Read done')
	for i, cat in enumerate(categories):
		ind = i + 1
		logthis.say(f'Train test split starts for {cat=} category {ind}/{len(categories)}')
		x_train = df_train[TEXT].astype('U')
		y_train = df_train[LABEL]
		x_test = df_test[TEXT].astype('U')
		y_test = df_test[LABEL]

		#undersample = RandomUnderSampler(sampling_strategy=get_sampling_strategy(df_train, categories, cat))
		undersample = RandomUnderSampler(sampling_strategy='majority')
		x_train, y_train = undersample.fit_resample(x_train.to_frame(TEXT), y_train)
		get_sampling_strategy(y_train.to_frame(LABEL), categories, cat)
		logthis.say(f'Filter starts for {cat=} category {ind}/{len(categories)}')
		y_train = y_train.to_frame(LABEL)
		filter_dataframe(y_train, cat)
		y_test = y_test.to_frame(LABEL)
		filter_dataframe(y_test, cat)
		y_train = np.ravel(y_train)
		y_test = np.ravel(y_test)
		logthis.say(f'Filtering done')

		result_storage = ResultStorage(train, test, cat, evaluation_metric)

		logthis.say(f'Logistic regression starts for {cat=} category {ind}/{len(categories)}')
		pipeline = Pipeline([
			('tfidf', TfidfVectorizer(max_df=1.0, min_df=0.0, max_features=None, ngram_range=(1, 2))),
            ('lr', LogisticRegression(max_iter=90000, random_state=1))])
		param_grid = {
			'lr__penalty': ['l1', 'l2'],
			'lr__C': [1, 2, 4, 5, 7, 9, 10]
		}
		logthis.say(pipeline.get_params().keys())
		search = GridSearchCV(pipeline, param_grid, n_jobs=-1, verbose=2, error_score=-1, return_train_score=True, refit='f1_weighted', scoring=['accuracy', 'f1_macro', 'f1_weighted', 'recall_macro', 'recall_weighted', 'precision_macro', 'precision_weighted'])
		search.fit(x_train[TEXT], y_train)
		logthis.say("Best parameter (CV score=%0.3f):" % search.best_score_)
		logthis.say(search.best_params_)
		df = pd.DataFrame(search.cv_results_)
		df.to_csv('data/search/model_lr_search_'+ cat + '.csv', sep=';')

		logthis.say(f'SVC starts for {cat=} category {ind}/{len(categories)}')
		pipeline = Pipeline([
			('tfidf', TfidfVectorizer(max_df=1.0, min_df=0.0, max_features=None, ngram_range=(1, 2))),
            ('svc', SVC(probability=True))])
		param_grid = {
			'svc__C':[1,10,100,1000],
			'svc__gamma':[1,0.1,0.001,0.0001], 
			'svc__kernel':['linear','rbf']
		}
		logthis.say(pipeline.get_params().keys())
		search = GridSearchCV(pipeline, param_grid, n_jobs=-1, verbose=2, error_score=-1, return_train_score=True, refit='f1_weighted', scoring=['accuracy', 'f1_macro', 'f1_weighted', 'recall_macro', 'recall_weighted', 'precision_macro', 'precision_weighted'])
		search.fit(x_train[TEXT], y_train)
		logthis.say("Best parameter (CV score=%0.3f):" % search.best_score_)
		logthis.say(search.best_params_)
		df = pd.DataFrame(search.cv_results_)
		df.to_csv('data/search/model_svc_search_'+ cat + '.csv', sep=';')

		logthis.say(f'KNeighborsClassifier starts for {cat=} category {ind}/{len(categories)}')
		pipeline = Pipeline([
			('tfidf', TfidfVectorizer(max_df=1.0, min_df=0.0, max_features=None, ngram_range=(1, 2))),
            ('knn', KNeighborsClassifier(n_neighbors=20, weights='distance', n_jobs=-1))])
		param_grid = {
			'knn__n_neighbors': (1,10,20, 30),
			'knn__leaf_size': (20,40,1),
			'knn__p': (1,2),
			'knn__weights': ('uniform', 'distance'),
			'knn__metric': ('minkowski', 'chebyshev')
		}
		logthis.say(pipeline.get_params().keys())
		search = GridSearchCV(pipeline, param_grid, n_jobs=-1, verbose=2, error_score=-1, return_train_score=True, refit='f1_weighted', scoring=['accuracy', 'f1_macro', 'f1_weighted', 'recall_macro', 'recall_weighted', 'precision_macro', 'precision_weighted'])
		search.fit(x_train[TEXT], y_train)
		logthis.say("Best parameter (CV score=%0.3f):" % search.best_score_)
		logthis.say(search.best_params_)
		df = pd.DataFrame(search.cv_results_)
		df.to_csv('data/search/model_knn_search_'+ cat + '.csv', sep=';')


		logthis.say(f'RandomForestClassifier starts for {cat=} category {ind}/{len(categories)}')
		pipeline = Pipeline([
			('tfidf', TfidfVectorizer(max_df=1.0, min_df=0.0, max_features=None, ngram_range=(1, 2))),
            ('randomforest', RandomForestClassifier(bootstrap=True,min_impurity_decrease=1e-7,n_jobs=-1, random_state=42))])
		param_grid = {
			'randomforest__n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 2)],
            'randomforest__max_features': ['auto', 'sqrt'],
            'randomforest__max_depth': [int(x) for x in np.linspace(10, 100, num = 2)]+[None],
            'randomforest__min_samples_split': [2, 5, 10],
            'randomforest__min_samples_leaf': [1, 2, 4],
            'randomforest__bootstrap': [True, False]
		}		
		logthis.say(pipeline.get_params().keys())
		search = GridSearchCV(pipeline, param_grid, n_jobs=-1, verbose=2, error_score=-1, return_train_score=True, refit='f1_weighted', scoring=['accuracy', 'f1_macro', 'f1_weighted', 'recall_macro', 'recall_weighted', 'precision_macro', 'precision_weighted'])
		search.fit(x_train[TEXT], y_train)
		logthis.say("Best parameter (CV score=%0.3f):" % search.best_score_)
		logthis.say(search.best_params_)
		df = pd.DataFrame(search.cv_results_)
		df.to_csv('data/search/model_randomforest_search_'+ cat + '.csv', sep=';')


		logthis.say(f'LinearSVC starts for {cat=} category {ind}/{len(categories)}')
		pipeline = Pipeline([
			('tfidf', TfidfVectorizer(max_df=1.0, min_df=0.0, max_features=None, ngram_range=(1, 2))),
            ('linearSVC', LinearSVC()	)])
		param_grid = {
			'linearSVC__base_estimator__C': [0.00001, 0.0001, 0.0005],
            'linearSVC__base_estimator__dual': (True, False), 
			'linearSVC__base_estimator__random_state': [1]
		}
		logthis.say(pipeline.get_params().keys())
		search = GridSearchCV(pipeline, param_grid, n_jobs=-1, verbose=2, error_score=-1, return_train_score=True, refit='f1_weighted', scoring=['accuracy', 'f1_macro', 'f1_weighted', 'recall_macro', 'recall_weighted', 'precision_macro', 'precision_weighted'])
		search.fit(x_train[TEXT], y_train)
		logthis.say("Best parameter (CV score=%0.3f):" % search.best_score_)
		logthis.say(search.best_params_)
		df = pd.DataFrame(search.cv_results_)
		df.to_csv('data/search/model_linearsvc_search_'+ cat + '.csv', sep=';')

if __name__ == "__main__":
	train_models('data/train_test_data/readme_new_preprocessed_train.csv', 'data/train_test_data/readme_new_preprocessed_test.csv', 'data/search/', 'search_model.csv', ["Natural Language Processing", "Computer Vision", "Sequential", "Audio", "Graphs", "Reinforcement Learning"])
