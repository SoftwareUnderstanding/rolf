from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def getWordLevelVectorizer(df : pd.DataFrame, textcolname : str) -> TfidfVectorizer:
	# word level tf-idf
	tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=10000)
	tfidf_vect.fit(df[textcolname])
	print("word level tf-idf done")
	return tfidf_vect

def getNGramLevelVectorizer(df : pd.DataFrame, textcolname : str) -> TfidfVectorizer:
	# ngram level tf-idf 
	tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=10000)
	tfidf_vect_ngram.fit(df[textcolname])
	print("ngram level tf-idf done")
	return tfidf_vect_ngram


def getCharLevelVectorizer(df : pd.DataFrame, textcolname : str) -> TfidfVectorizer:
	# characters level tf-idf
	tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char',  ngram_range=(2,3), max_features=10000) #token_pattern=r'\w{1,}',
	tfidf_vect_ngram_chars.fit(df[textcolname])
	print("characters level tf-idf done")
	return tfidf_vect_ngram_chars