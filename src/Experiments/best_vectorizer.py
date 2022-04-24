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
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import confusion_matrix

#from src.ResultStorage import ResultStorage

TEXT = "Text"
LABEL = "Label"

tn = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[0, 0]
fp = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[0, 1]
fn = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[1, 0]
tp = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[1, 1]

def filter_dataframe(df, cat):
	#count = 0
	for ind, row in df.iterrows():
		if cat != str(row[LABEL]):
			#count += 1
			row[LABEL] = 'Other'	
	#print(f'{cat} filtered {count} rows in training dataset')

def train_models(train: str, categories: List[str] = None) -> None:
	if categories is None:
		categories = ["Natural Language Processing", "Computer Vision", "Sequential", "Audio", "Graphs", "Reinforcement Learning"]
	logthis.say(f'Read files\nTrain dataset: {train} \n')
	df_train = pd.read_csv(train, sep=';')
	#df_test.drop_duplicates(subset=['Text'], inplace=True, keep='last')
	df_train = df_train.drop(columns = 'Repo')
	logthis.say('Read done')
	for i, cat in enumerate(categories):
		#result_storage = ResultStorage(train, cat, evaluation_metric)

		#undersample = RandomUnderSampler(sampling_strategy=get_sampling_strategy(df_train, categories, cat))
		#undersample = RandomUnderSampler(sampling_strategy='majority')
		#x_train, y_train = undersample.fit_resample(x_train.to_frame(TEXT), y_train)

		ind = i + 1
		logthis.say(f'Train test split starts for {cat=} category {ind}/{len(categories)}')
		x_train = df_train[TEXT].astype('U')
		y_train = df_train[LABEL]

		logthis.say(f'Filter starts for {cat=} category {ind}/{len(categories)}')
		y_train = y_train.to_frame(LABEL)
		filter_dataframe(y_train, cat)
		y_train = np.ravel(y_train)
		logthis.say(f'Filtering done')

		undersample = RandomUnderSampler(sampling_strategy='majority')
		x_train, y_train = undersample.fit_resample(x_train.to_frame(TEXT), y_train)

		#countvect = CountVectorizer(analyzer="word",token_pattern=r'\w{1,}', max_features=10000, lowercase=True)
		#tfidf = TfidfVectorizer(analyzer="word",token_pattern=r'\w{1,}', max_features=10000, lowercase=True)

		pipeline = Pipeline(
			[
				("vect", TfidfVectorizer()),
				("clf", LinearSVC(max_iter=900000)),
			]
		)
		parameters = {
			"vect__max_df": (1.0, 0.9, 0.8),
			#"vect__max_df": [1.0],
			"vect__min_df": (0.1, 0.15, 0.2),
			#"vect__min_df": [0.0],
			'vect__max_features': (None, 1500, 2000, 2500, 3000, 5000, 10000),
			#'vect__max_features' : (1500, 2000)
			"vect__ngram_range": ((1, 1), (1, 2)),  # unigrams or bigrams
		}

		search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=2, error_score=-1, return_train_score=True, refit='f1_weighted', scoring=['accuracy', 'f1_macro', 'f1_weighted', 'recall_macro', 'recall_weighted', 'precision_macro', 'precision_weighted'])
		search.fit(x_train[TEXT], y_train)
		print("Best parameter (CV score=%0.3f):" % search.best_score_)
		print(search.best_params_)
		df = pd.DataFrame(search.cv_results_)
		df.to_csv('data/search/vectorizer_search_tfidf_majority_negative_samples'+ '_'.join(cat.lower().split(' ')) + '.csv', sep=';')

if __name__ == "__main__":
	train_models('data/train_test_data/readme_negative_samples_preprocessed_train.csv')
