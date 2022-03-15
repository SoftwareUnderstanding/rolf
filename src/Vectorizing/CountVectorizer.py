from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

def getCountVectorizer(df : pd.DataFrame, textcolname : str):
	count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=10000)
	count_vect.fit(df[textcolname])
	print("count vectorizer done")
	return count_vect
