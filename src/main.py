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
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from ResultStorage import ResultStorage
from tqdm import tqdm
import fasttext
from imblearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from Vectorizing.CountVectorizer import getCountVectorizer
from Vectorizing.TF_IDF_Vectorizer import getWordLevelVectorizer, getNGramLevelVectorizer
from Embedding.WordEmbedding import createWordEmbedding
from Report import Report
from Report.CrossValidateNN import cross_validate_NN
from lstmModel import create_lstm_model, create_bidirec_lstm_model
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import CountVectorizer

TEXT = "Text"
LABEL = "Label"
CV_splits = 5
sample = True
nb_sample = 5000

#datasets = [('abstracts_train.csv', 'abstracts_test.csv'),
#			('somef_data_train.csv', 'somef_data_test.csv'),
#			('somef_data_description_train.csv', 'somef_data_description_test.csv'),
#			('merged_abstracts_somef_data_description.csv', 'somef_data.csv'),
#			('merged_abstracts_somef_data.csv', 'somef_data_description.csv'),
#			('merged_somef_data_somef_data_description.csv', 'abstracts.csv')]

datasets = [#('readme_semantic_web_train.csv', 'readme_semantic_web_test.csv')]
			#('somef_data_train.csv', 'somef_data_test.csv'),
			('readme_train.csv', 'readme_test.csv')]
			#('abstracts.csv', 'somef_data.csv'),
			#('readme.csv', 'somef_data.csv'),
			#('somef_data.csv', 'readme.csv'),
			#('abstracts.csv', 'readme.csv'),
			#('merged_abstracts_train_somef_data_train.csv', 'somef_data_test.csv'),
			#('merged_abstracts_somef_data_train_readme_train.csv', 'somef_data_test.csv'),
			#('merged_abstracts_somef_data_train_readme_train.csv', 'readme_test.csv')]


def filter_dataframe(df, cat):
	count = 0
	for ind, row in df.iterrows():
		if cat != str(row[LABEL]):
			count += 1
			row[LABEL] = 'Other'
		
	print(f'{cat} filtered {count} rows in training dataset')

def get_sampling_strategy(df_train):
	sizes = df_train.groupby(LABEL).size()
	indexes = list(sizes.index)
	cat_size = sizes[indexes.index(cat)]
	other_cat_size = int(cat_size/(len(df_train[LABEL].unique())-2))+1
	sampling_stratgy = {}
	for c in df_train[LABEL].unique():
		if c == cat:
			sampling_stratgy[c] = cat_size
		elif c == 'General':
			sampling_stratgy[c] = 0
		else:
			sampling_stratgy[c] = min(other_cat_size, sizes[indexes.index(c)])
	print('Sampling strategy: ', sampling_stratgy)
	return sampling_stratgy


counter = 1
for train, test in datasets:
	categories = ["Sequential", "Natural Language Processing", "Audio", "Computer Vision", "Graphs", "Reinforcement Learning"]
	#categories = ['Sequential']

	for i, cat in enumerate(categories):
		print(f'Read file {counter}/{len(datasets)} \nTrain dataset: {train} \nTest dataset: {test}')
		start_time = time()
		df_train = pd.read_csv('data/'+train, sep=';')
		df_test = pd.read_csv('data/'+test, sep = ';')
		
		df_train.drop_duplicates(subset=['Text'], inplace=True, keep=False)
		df_train.drop( df_train[ df_train[TEXT] == "" ].index , inplace=True)
		df_train.drop( df_train[ df_train[LABEL] == "General" ].index , inplace=True)
		df_test.drop( df_test[ df_test[LABEL] == "General" ].index , inplace=True)
		df_test.drop( df_test[ df_test[TEXT] == "" ].index , inplace=True)
		

		ind = i + 1
		start_time = time()
		print(f'Train test split starts for {cat=} category {ind}/{len(categories)}')
		df_train = df_train.drop(columns = 'Repo')
		x_train = df_train[TEXT]
		y_train = df_train[LABEL]
		x_test = df_test[TEXT]
		y_test = df_test[LABEL]
		
		undersample = RandomUnderSampler(sampling_strategy=get_sampling_strategy(df_train))
		#undersample = RandomUnderSampler(sampling_strategy='majority')
		x_train, y_train = undersample.fit_resample(x_train.to_frame(TEXT), y_train)

		print(f'Filter starts for {cat=} category {ind}/{len(categories)}')
		y_train = y_train.to_frame(LABEL)
		filter_dataframe(y_train, cat)
		y_test = y_test.to_frame(LABEL)
		filter_dataframe(y_test, cat)
		y_train = np.ravel(y_train)
		y_test = np.ravel(y_test)
		print(f'Filtering done in: {time()-start_time:.2f} s')

		result_storage = ResultStorage(train, test, cat)

		print(f'Logistic regression starts for {cat=} category {ind}/{len(categories)}')
		pipeline = Pipeline([
			('tfidf', CountVectorizer(analyzer="word",token_pattern=r'\w{1,}', max_features=10000, lowercase=True)),
            ('lr', LogisticRegression(max_iter=10000, random_state=1))])
		result_storage.processResult(*Report.report(pipeline, train, test, x_train[TEXT], y_train, x_test, y_test, cat, name='LR_Count_Vectors_RandomUnder', cv=CV_splits, dict_scoring=Report.score_metrics, save=False))
		pipeline = Pipeline([
			('tfidf', TfidfVectorizer(analyzer="word",token_pattern=r'\w{1,}', max_features=10000, lowercase=True)),
            ('lr', LogisticRegression(max_iter=10000, random_state=1))])
		result_storage.processResult(*Report.report(pipeline, train, test, x_train[TEXT], y_train, x_test, y_test, cat, name='LR_TFIDF_RandomUnder', cv=CV_splits, dict_scoring=Report.score_metrics, save=False))
		
		print(f'SVC starts for {cat=} category {ind}/{len(categories)}')
		pipeline = Pipeline([
			('tfidf', CountVectorizer(analyzer="word",token_pattern=r'\w{1,}', max_features=10000, lowercase=True)),
            ('lr', SVC(probability=True))])
		result_storage.processResult(*Report.report(pipeline, train, test, x_train[TEXT], y_train, x_test, y_test, cat, name='SVC_Count_Vectors_RandomUnder', cv=CV_splits, dict_scoring=Report.score_metrics, save=False))		
		pipeline = Pipeline([
			('tfidf', TfidfVectorizer(analyzer="word",token_pattern=r'\w{1,}', max_features=10000, lowercase=True)),
            ('lr', SVC(probability=True))])
		result_storage.processResult(*Report.report(pipeline, train, test, x_train[TEXT], y_train, x_test, y_test, cat, name='SVC_TFIDF_RandomUnder', cv=CV_splits, dict_scoring=Report.score_metrics, save=False))
		
		print(f'KNeighborsClassifier starts for {cat=} category {ind}/{len(categories)}')
		pipeline = Pipeline([
			('tfidf', CountVectorizer(analyzer="word",token_pattern=r'\w{1,}', max_features=10000, lowercase=True)),
            ('lr', KNeighborsClassifier(n_neighbors=20, weights='distance', n_jobs=-1))])
		result_storage.processResult(*Report.report(pipeline, train, test, x_train[TEXT], y_train, x_test, y_test, cat, name='KNN_Count_Vectors_RandomUnder', cv=CV_splits, dict_scoring=Report.score_metrics, save=False))
		pipeline = Pipeline([
			('tfidf', TfidfVectorizer(analyzer="word",token_pattern=r'\w{1,}', max_features=10000, lowercase=True)),
            ('lr', KNeighborsClassifier(n_neighbors=20, weights='distance', n_jobs=-1))])
		result_storage.processResult(*Report.report(pipeline, train, test, x_train[TEXT], y_train, x_test, y_test, cat, name='KNN_TFIDF_RandomUnder', cv=CV_splits, dict_scoring=Report.score_metrics, save=False))
		
		print(f'RandomForestClassifier starts for {cat=} category {ind}/{len(categories)}')
		pipeline = Pipeline([
			('tfidf', CountVectorizer(analyzer="word",token_pattern=r'\w{1,}', max_features=10000, lowercase=True)),
            ('lr', RandomForestClassifier(bootstrap=True,min_impurity_decrease=1e-7,n_jobs=-1, random_state=42))])
		result_storage.processResult(*Report.report(pipeline, train, test, x_train[TEXT], y_train, x_test, y_test, cat, name='RandomForestClassifier_Count_Vectors_RandomUnder', cv=CV_splits, dict_scoring=Report.score_metrics, save=False))
		pipeline = Pipeline([
			('tfidf', TfidfVectorizer(analyzer="word",token_pattern=r'\w{1,}', max_features=10000, lowercase=True)),
            ('lr', RandomForestClassifier(bootstrap=True,min_impurity_decrease=1e-7,n_jobs=-1, random_state=42))])
		result_storage.processResult(*Report.report(pipeline, train, test, x_train[TEXT], y_train, x_test, y_test, cat, name='RandomForestClassifier_TFIDF_RandomUnder', cv=CV_splits, dict_scoring=Report.score_metrics, save=False))
		

		print(f'LinearSVC starts for {cat=} category {ind}/{len(categories)}')
		pipeline = Pipeline([
			('tfidf', CountVectorizer(analyzer="word",token_pattern=r'\w{1,}', max_features=10000, lowercase=True)),
            ('lr', CalibratedClassifierCV(LinearSVC()))])
		result_storage.processResult(*Report.report(pipeline, train, test, x_train[TEXT], y_train, x_test, y_test, cat, name='Linear_SVC_Count_Vectors_RandomUnder', cv=CV_splits, dict_scoring=Report.score_metrics, save=False))
		pipeline = Pipeline([
			('tfidf', TfidfVectorizer(analyzer="word",token_pattern=r'\w{1,}', max_features=10000, lowercase=True)),
            ('lr', CalibratedClassifierCV(LinearSVC()))])
		result_storage.processResult(*Report.report(pipeline, train, test, x_train[TEXT], y_train, x_test, y_test, cat, name='Linear_SVC_TFIDF_RandomUnder', cv=CV_splits, dict_scoring=Report.score_metrics, save=False))
	

		result_storage.dumpBestModel()
	
		result_storage.dumpResults()
	