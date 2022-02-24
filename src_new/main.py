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
from tqdm import tqdm
import fasttext

from Vectorizing.CountVectorizer import getCountVectorizer
from Vectorizing.TF_IDF_Vectorizer import getWordLevelVectorizer
from Embedding.WordEmbedding import createWordEmbedding
from Report import Report
from writeResults import writeResults
from Report.CrossValidateNN import cross_validate_NN
from lstmModel import create_lstm_model, create_bidirec_lstm_model

TEXT = "clean_Text"
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

datasets = [('merged_abstracts_somef_data_train_readme_train.csv', 'readme_test.csv'),
			('abstracts.csv', 'readme_test.csv'),
			('somef_data_train.csv', 'readme_test.csv'),
			('readme_train.csv', 'readme_test.csv'),
			('merged_abstracts_somef_data_train_readme_train.csv', 'somef_data_test.csv'),
			('abstracts.csv', 'somef_data_test.csv'),
			('somef_data_train.csv', 'somef_data_test.csv'),
			('readme_train.csv', 'somef_data_test.csv')]

results_filename = 'results.csv'
df_results = pd.DataFrame()

def filter_dataframe(df, cat):
	count = 0
	for ind, row in df.iterrows():
		if cat != str(row[LABEL]):
			count += 1
			row[LABEL] = 'Other'
	print(f'{cat} filtered {count} rows in training dataset')

counter = 1
for train, test in datasets:
	
	print(f'Reaf file {counter}/{len(datasets)} \nTrain dataset: {train} \nTest dataset: {test}')
	start_time = time()
	df_train = pd.read_csv('data/'+train, sep=';')
	df_test = pd.read_csv('data/'+test, sep = ';')
	
	print(f'Read done in: {time()-start_time:.2f} s')
	start_time = time()
	print(f'Start preprocessor')
	Preprocessor(df_train).run()
	Preprocessor(df_test).run()
	print(f'Preprocessing done in: {time()-start_time:.2f} s')
	start_time = time()
	#print(df_test.head())

	categories = df_train[LABEL].unique()
	#categories = ['Audio']

	for i, cat in enumerate(categories):
		ind = i + 1
		print(f'Filter starts for {cat=} category {ind}/{len(categories)}')
		filter_dataframe(df_train, cat)
		filter_dataframe(df_test, cat)
		print(f'Filering done in: {time()-start_time:.2f} s')
		
		start_time = time()
		print(f'Train test split starts for {cat=} category {ind}/{len(categories)}')
		df_train = df_train.drop(columns = 'Text')
		df_train = df_train.drop(columns = 'Repo')
		x_train = df_train[TEXT]
		y_train = df_train[LABEL]
		x_test = df_test[TEXT]
		y_test = df_test[LABEL]

		encoder = LabelEncoder()
		y_train = encoder.fit_transform(y_train)
		y_test = encoder.transform(y_test)
		print('y_test: ', y_test)
		print(f'Train test split done in: {time()-start_time:.2f} s')

		start_time = time()
		# CountVecotirzing
		print(f'Count vectorizer starts for {cat=} category {ind}/{len(categories)}')
		count_vect = getCountVectorizer(df_train, TEXT)
		xtrain_count = count_vect.transform(x_train)
		xtest_count = count_vect.transform(x_test)
		print(f'Count vectorizing done in: {time()-start_time:.2f} s')

		start_time = time()
		# TF-IDF
		print(f'TF-IDF starts for {cat=} category {ind}/{len(categories)}')
		count_vect = getWordLevelVectorizer(df_train, TEXT)
		xtrain_tfidf = count_vect.transform(x_train)
		xtest_tfidf = count_vect.transform(x_test)
		print(f'TF-IDF done in: {time()-start_time:.2f} s')

		#start_time = time()
		#print(f'Word embedding starts for {cat=} category {ind}/{len(categories)}')
		#word_index, xtrain_embeddingmatrix = createWordEmbedding(df_train, TEXT)
		#token = Tokenizer(oov_token='<OOV>')
		#token.fit_on_texts(df_train[TEXT])
		#train_seq_x = sequence.pad_sequences(token.texts_to_sequences(x_train), maxlen=300)
		#test_seq_x = sequence.pad_sequences(token.texts_to_sequences(x_test), maxlen=300)
		
		#token = Tokenizer(oov_token='<OOV>')
		#token.fit_on_texts(df_train[TEXT])
		#word_index = token.word_index
		# convert text to sequence of tokens and pad them to ensure equal length vectors 
		#train_seq_x = sequence.pad_sequences(token.texts_to_sequences(x_train), maxlen=300)
		#test_seq_x = sequence.pad_sequences(token.texts_to_sequences(x_test), maxlen=300)

		# create token-embedding mapping
		#!wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip
    	#!unzip crawl-300d-2M-subword.zip
		#pretrained = fasttext.FastText.load_model('src_new/crawl-300d-2M-subword.bin')
		#embedding_matrix = np.zeros((len(word_index) + 1, 300))
		#words = []
		#for word, i in tqdm(word_index.items()):
		#	embedding_vector = pretrained.get_word_vector(word) #embeddings_index.get(word)
		#	words.append(word)
		#	if embedding_vector is not None:
		#		embedding_matrix[i] = embedding_vector
		#print(f'Word embedding done in: {time()-start_time:.2f} s')

		start_time = time()
		print(f'Random undersampler starts for {cat=} category {ind}/{len(categories)}')
		print(f'Shapes befor undersampler: xtrain_count: {xtrain_count.shape}, xtest_count: {xtest_count.shape}')
		print('Train label: ', df_train.groupby(LABEL).size())
		print('Test labels: ', df_test.groupby(LABEL).size())
		undersample = RandomUnderSampler(sampling_strategy='majority')
		xtrain_count, ytrain_count = undersample.fit_resample(xtrain_count, y_train)
		xtest_count, ytest_count = undersample.fit_resample(xtest_count, y_test)
		#print('Train label: ', ytrain_count.value_counts())
		#print('Test labels: ', ytest_count.value_counts())
		print(f'Shapes after undersampler: xtrain_count: {xtrain_count.shape}, xtest_count: {xtest_count.shape}')
		print(f'Random undersampler done in: {time()-start_time:.2f} s')

		start_time = time()
		print(f'Random undersampler starts for {cat=} category {ind}/{len(categories)}')
		print(f'Shapes befor undersampler: xtrain_tfidf: {xtrain_tfidf.shape}, xtest_tfidf: {xtest_tfidf.shape}')
		print('Train label: ', df_train.groupby(LABEL).size())
		print('Test labels: ', df_test.groupby(LABEL).size())
		xtrain_tfidf, ytrain_tfidf = undersample.fit_resample(xtrain_tfidf, y_train)
		xtest_tfidf, ytest_tfidf = undersample.fit_resample(xtest_tfidf, y_test)
		#print('Train label: ', ytrain_tfidf.value_counts())
		#print('Test labels: ', ytest_tfidf.value_counts())
		print(f'Shapes after undersampler: xtrain_tfidf: {xtrain_tfidf.shape}, xtest_tfidf: {xtest_tfidf.shape}')
		print(f'Random undersampler done in: {time()-start_time:.2f} s')

		start_time = time()
		print(f'Label encoder starts for {cat=} category {ind}/{len(categories)}')
		encoder = LabelEncoder()
		trainy_count = encoder.fit_transform(ytrain_count)
		testy_count = encoder.transform(ytest_count)

		trainy_tfidf = encoder.fit_transform(ytrain_tfidf)
		testy_tfidf = encoder.transform(ytest_tfidf)

		labels = [0, 1]
		test1=pd.DataFrame(data=np.transpose([labels,encoder.transform(labels)]), columns=["labels", "encoding"]).sort_values(by=["encoding"])
		labels=test1.labels.tolist()
		print(f'Labels: {labels}')
		if any([0,1]) in labels and len(labels)==2:
			labels[labels.index(0)] = "negative"
			labels[labels.index(1)] = "positive"
		print(f'Labels: {labels}')
		print(f'Label encoder done in: {time()-start_time:.2f} s')

		#print(f'LSTM starts for {cat=} category {ind}/{len(categories)}')
		#df_results = df_results.append(cross_validate_NN(create_lstm_model(word_index, labels, embedding_matrix, pre_trained=False), 
        #                                             train_seq_x, y_train, test_seq_x, y_test, name="LSTM_WE",
        #                                             scoring=Report.score_metrics, n_splits=CV_splits, save=True))
		#writeResults('results.csv', df_results, train, test)

		#print(f'Bi-LSTM starts for {cat=} category {ind}/{len(categories)}')
		#df_results = df_results.append(cross_validate_NN(create_bidirec_lstm_model(word_index, labels, embedding_matrix, pre_trained=False), 
        #                                             train_seq_x, y_train, test_seq_x, y_test, name="Bi-LSTM_WE",
        #                                             scoring=Report.score_metrics, n_splits=CV_splits, save=True))
		#writeResults('results.csv', df_results, train, test)

		print(f'Logistic regression starts for {cat=} category {ind}/{len(categories)}')
		df_results = df_results.append(Report.report(LogisticRegression(max_iter=1000), xtrain_count, trainy_count, xtest_count, testy_count, cat, name='LR_Count_Vectors', cv=CV_splits, dict_scoring=Report.score_metrics, save=False))
		writeResults('results.csv', df_results, train, test)
		df_results = df_results.append(Report.report(LogisticRegression(max_iter=1000), xtrain_tfidf, trainy_tfidf, xtest_tfidf, testy_tfidf, cat, name='LR_TFIDF', cv=CV_splits, dict_scoring=Report.score_metrics, save=False))
		writeResults('results.csv', df_results, train, test)
		print(df_results)

		print(f'SVC starts for {cat=} category {ind}/{len(categories)}')
		df_results = df_results.append(Report.report(SVC(), xtrain_count, trainy_count, xtest_count, testy_count, cat, name='SVC_Count_Vectors', cv=CV_splits, dict_scoring=Report.score_metrics, save=False))
		writeResults('results.csv', df_results, train, test)
		df_results = df_results.append(Report.report(SVC(), xtrain_tfidf, trainy_tfidf, xtest_tfidf, testy_tfidf, cat, name='SVC_TFIDF', cv=CV_splits, dict_scoring=Report.score_metrics, save=False))
		writeResults('results.csv', df_results, train, test)
		print(df_results)

		print(f'KNeighborsClassifier starts for {cat=} category {ind}/{len(categories)}')
		df_results = df_results.append(Report.report(KNeighborsClassifier(n_neighbors=20, weights='distance', n_jobs=-1), xtrain_count, trainy_count, xtest_count, testy_count, cat, name='KNN_Count_Vectors', cv=CV_splits, dict_scoring=Report.score_metrics, save=False))
		writeResults('results.csv', df_results, train, test)
		df_results = df_results.append(Report.report(KNeighborsClassifier(n_neighbors=20, weights='distance', n_jobs=-1), xtrain_tfidf, trainy_tfidf, xtest_tfidf, testy_tfidf, cat, name='KNN_TFIDF', cv=CV_splits, dict_scoring=Report.score_metrics, save=False))
		writeResults('results.csv', df_results, train, test)
		print(df_results)
		
		print(f'RandomForestClassifier starts for {cat=} category {ind}/{len(categories)}')
		df_results = df_results.append(Report.report(RandomForestClassifier(bootstrap=True,min_impurity_decrease=1e-7,n_jobs=-1, random_state=42), xtrain_count, trainy_count, xtest_count, testy_count, cat, name='RandomForestClassifier_Count_Vectors', cv=CV_splits, dict_scoring=Report.score_metrics, save=False))
		writeResults('results.csv', df_results, train, test)
		df_results = df_results.append(Report.report(RandomForestClassifier(bootstrap=True,min_impurity_decrease=1e-7,n_jobs=-1, random_state=42), xtrain_tfidf, trainy_tfidf, xtest_tfidf, testy_tfidf, cat, name='RandomForestClassifier_TFIDF', cv=CV_splits, dict_scoring=Report.score_metrics, save=False))
		writeResults('results.csv', df_results, train, test)
		print(df_results)

		print(f'SGDClassifier starts for {cat=} category {ind}/{len(categories)}')
		df_results = df_results.append(Report.report(SGDClassifier(loss='modified_huber', max_iter=1000, tol=1e-3,   n_iter_no_change=10, early_stopping=True, n_jobs=-1 ), xtrain_count, trainy_count, xtest_count, testy_count, cat, name='SGDClassifier_Count_Vectors', cv=CV_splits, dict_scoring=Report.score_metrics, save=False))
		writeResults('results.csv', df_results, train, test)
		df_results = df_results.append(Report.report(SGDClassifier(loss='modified_huber', max_iter=1000, tol=1e-3,   n_iter_no_change=10, early_stopping=True, n_jobs=-1 ), xtrain_tfidf, trainy_tfidf, xtest_tfidf, testy_tfidf, cat, name='SGDClassifier_TFIDF', cv=CV_splits, dict_scoring=Report.score_metrics, save=False))
		writeResults('results.csv', df_results, train, test)
		print(df_results)

		print(f'AdaBoostClassifier starts for {cat=} category {ind}/{len(categories)}')
		df_results = df_results.append(Report.report(AdaBoostClassifier(n_estimators=1000), xtrain_count, trainy_count, xtest_count, testy_count, cat, name='AdaBoostClassifier_Count_Vectors', cv=CV_splits, dict_scoring=Report.score_metrics, save=False))
		writeResults('results.csv', df_results, train, test)
		df_results = df_results.append(Report.report(AdaBoostClassifier(n_estimators=1000), xtrain_tfidf, trainy_tfidf, xtest_tfidf, testy_tfidf, cat, name='AdaBoostClassifier_TFIDF', cv=CV_splits, dict_scoring=Report.score_metrics, save=False))
		writeResults('results.csv', df_results, train, test)
		print(df_results)

		print(f'LinearSVC starts for {cat=} category {ind}/{len(categories)}')
		df_results = df_results.append(Report.report(LinearSVC(), xtrain_count, trainy_count, xtest_count, testy_count, cat, name='SVC_Count_Vectors', cv=CV_splits, dict_scoring=Report.score_metrics, save=False))
		writeResults('results.csv', df_results, train, test)
		df_results = df_results.append(Report.report(LinearSVC(), xtrain_tfidf, trainy_tfidf, xtest_tfidf, testy_tfidf, cat, name='SVC_TFIDF', cv=CV_splits, dict_scoring=Report.score_metrics, save=False))
		writeResults('results.csv', df_results, train, test)
		print(df_results)

		print(f'MultinomialNB starts for {cat=} category {ind}/{len(categories)}')
		df_results = df_results.append(Report.report(MultinomialNB(), xtrain_count, trainy_count, xtest_count, testy_count, cat, name='SVC_Count_Vectors', cv=CV_splits, dict_scoring=Report.score_metrics, save=False))
		writeResults('results.csv', df_results, train, test)
		df_results = df_results.append(Report.report(MultinomialNB(), xtrain_tfidf, trainy_tfidf, xtest_tfidf, testy_tfidf, cat, name='SVC_TFIDF', cv=CV_splits, dict_scoring=Report.score_metrics, save=False))
		writeResults('results.csv', df_results, train, test)
		print(df_results)

		df_results.to_csv('final_results.csv', sep=';')






