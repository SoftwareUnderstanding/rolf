#from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import logthis
import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import cross_validate


TEXT = "Text"
LABEL = "Label"

def filter_dataframe(df, cat):
	for ind, row in df.iterrows():
		if cat != str(row[LABEL]):
			row[LABEL] = 'Other'

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

		pipeline = Pipeline(
			[	
				("vect", TfidfVectorizer(max_df=1.0, min_df=0.0, max_features=None, ngram_range=(1, 2))),
				("clf_structured", LinearSVC(max_iter=900000)),
			]
		)

		scores = cross_validate(pipeline, x_train[TEXT], y_train, scoring=['accuracy', 'f1_macro', 'f1_weighted', 'recall_macro', 'recall_weighted', 'precision_macro', 'precision_weighted'],
			cv=5, return_train_score=True, n_jobs=-1, return_estimator=True)
		print('Scores: ', scores)
		df = pd.DataFrame(scores)

		x_train = df_train[TEXT].astype('U')
		y_train = df_train[LABEL]

		#undersample = RandomUnderSampler(sampling_strategy=get_sampling_strategy(df_train, categories, cat))
		undersample = RandomUnderSampler(sampling_strategy='majority')
		x_train, y_train = undersample.fit_resample(x_train.to_frame(TEXT), y_train)

		logthis.say(f'Filter starts for {cat=} category {ind}/{len(categories)}')
		y_train = y_train.to_frame(LABEL)
		filter_dataframe(y_train, cat)
		y_train = np.ravel(y_train)
		logthis.say(f'Filtering done')

		pipeline = Pipeline(
			[	
				("vect", TfidfVectorizer(max_df=1.0, min_df=0.0, max_features=None, ngram_range=(1, 2))),
				("clf_majority", LinearSVC(max_iter=900000)),
			]
		)

		scores = cross_validate(pipeline, x_train[TEXT], y_train, scoring=['accuracy', 'f1_macro', 'f1_weighted', 'recall_macro', 'recall_weighted', 'precision_macro', 'precision_weighted'],
			cv=5, return_train_score=True, n_jobs=-1, return_estimator=True)
		df1 = pd.DataFrame(scores)
		df = df.append(df1)
		#print(df)
		df.to_csv('data/search/sampler_search_'+ cat + '.csv', sep=';')

train_models('data/readme_new_preprocessed_train.csv', 'data/readme_new_preprocessed_test.csv')
