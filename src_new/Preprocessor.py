from os import remove
import pandas as pd
from keras.preprocessing.text import text_to_word_sequence
from numpy import loadtxt
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import re


TEXT = 'Text'

class Preprocessor:

	def __init__(self, data : pd.DataFrame) -> None:
		self.data = data

	def remove_stop_words(self, text : str, stop_words):
		'''
		Function to remove a list of words
		@param x : (str) text 
		@param stop_word: (list) list of stopwords to delete 
		@return: (str) new string without stopwords 
		'''

		token_list = text_to_word_sequence(text)	# tokenize text 
		return [token for token in token_list if token not in stop_words]
		
	def remove_punctuation(self, text):
		punctuationfree="".join([i for i in text if i not in string.punctuation])
		return punctuationfree

	def stemming(self, text, porter_stemmer):
		stem_text = [porter_stemmer.stem(word) for word in text]
		return stem_text
	
	def lemmatizer(self, text, wordnet_lemmatizer):
		lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
		return lemm_text
	
	def remove_links(self, text):
		regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
		return (re.sub(regex, '', text))

	def remove_links2(self, text):
		return [token for token in text if 'http' not in token]

	def run(self):
		NEWCOLNAME = TEXT
		#TODO: remove links
		#self.data[NEWCOLNAME]= self.data[TEXT].apply(lambda x: self.remove_links(x)) -> too slow

		#Remove punctuation
		self.data[NEWCOLNAME]= self.data[TEXT].apply(lambda x: self.remove_punctuation(x))

		#Transfor to lowercase
		self.data[NEWCOLNAME] = self.data[NEWCOLNAME].apply(lambda x: x.lower())

		#Remove stop words
		stop_words = stopwords.words('english')
		self.data[NEWCOLNAME] = self.data[NEWCOLNAME].apply(lambda x : self.remove_stop_words(x, stop_words))

		#Remove links
		self.data[NEWCOLNAME] = self.data[NEWCOLNAME].apply(lambda x : self.remove_links2(x))

		#Stemming
		wordnet_lemmatizer = WordNetLemmatizer()
		self.data[NEWCOLNAME]=self.data[NEWCOLNAME].apply(lambda x: self.lemmatizer(x, wordnet_lemmatizer))

		#Join tokens
		self.data[NEWCOLNAME]=self.data[NEWCOLNAME].apply(lambda x: ' '.join(x))
		#print(self.data.head())