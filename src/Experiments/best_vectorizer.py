from typing import List
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import logthis
import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

from src.ResultStorage import ResultStorage

TEXT = "Text"
LABEL = "Label"

def filter_dataframe(df, cat):
	#count = 0
	for ind, row in df.iterrows():
		if cat != str(row[LABEL]):
			#count += 1
			row[LABEL] = 'Other'	
	#print(f'{cat} filtered {count} rows in training dataset')

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
		result_storage = ResultStorage(train, test, cat, evaluation_metric)

		ind = i + 1
		logthis.say(f'Train test split starts for {cat=} category {ind}/{len(categories)}')
		x_train = df_train[TEXT].astype('U')
		y_train = df_train[LABEL]
		x_test = df_test[TEXT].astype('U')
		y_test = df_test[LABEL]

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

		pipeline = Pipeline(
			[
				("vect", CountVectorizer()),
				("clf", LinearSVC(max_iter=900000)),
			]
		)
		parameters = {
			"vect__max_df": (0.80, 0.90, 1.0),
			#"vect__max_df": [1.0],
			"vect__min_df": (0.0, 0.1, 0.20),
			#"vect__min_df": [0.0],
			'vect__max_features': (None, 1500, 2000, 2500, 3000, 5000, 10000),
			#'vect__max_features' : (1500, 2000)
			"vect__ngram_range": ((1, 1), (1, 2)),  # unigrams or bigrams
		}

		search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=2, error_score=-1, return_train_score=True, refit='f1_weighted', scoring=['accuracy', 'f1_macro', 'f1_weighted', 'recall_macro', 'recall_weighted', 'precision_macro', 'precision_weighted'])
		search.fit(x_train, y_train)
		print("Best parameter (CV score=%0.3f):" % search.best_score_)
		print(search.best_params_)
		df = pd.DataFrame(search.cv_results_)
		df.to_csv('data/search/vectorizer_search_'+ cat + '.csv', sep=';')

if __name__ == "__main__":
	train_models('data/readme_new_preprocessed_train.csv', 'data/readme_new_preprocessed_test.csv')
