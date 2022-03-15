#from os import remove
import pandas as pd
#from keras.preprocessing.text import text_to_word_sequence
from numpy import loadtxt
import string
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import re


TEXT = 'Text'

# Words that only apear in one category
#common = ['audio', 'pip', 'sample', 'music', 'wavenet', 'lyric', 'valloss', 'speaker', 'speech', 'cuda', 'folder', 'average', 'opencv', 'darknet', 'coco', 'demo', 'create', 'iteration', 'video', 'window', 'graph', 'acc', 'convolutional', 'feature', 'node', 'embedding', 'datasets', '110m', '12heads', 'available', 'made', 'released', 'original', 'found', 'experiment', 'larger', 'add', 'representation', 'update', 'bidirectional', 'answer', 'large', 'information', 'finetuning', 'transformer', 'detail', 'usage', 'dev', '12layer', 'based', 'method', '128', 'chinese', 'nlp', 'small', 'library', 'google', 'token', 'length', 'corpus', 'stateoftheart', 'character', 'may', 'however', 'likely', 'cased', '2018', 'release', 'contextual', 'export', 'tokenization', 'classification', 'multilingual', '768hidden', 'vocabulary', 'bertlarge', 'per', 'able', 'hyperparameters', 'accuracy', 'whole', 'running', 'bertbase', 'import', 'compatible', 'two', 'recommended', 'system', 'mask', 'architecture', 'work', 'tpu', '512', 'wordpiece', 'runsquadpy', 'attention', 'important', 'place', 'maximum', 'john', 'shell', 'uncased', 'case', 'vector', 'mean', 'issue', 'question', 'environment', 'agent', 'state', 'reward', 'action', 'episode', 'landmark', 'game', 'reinforcement', 'policy', 'qrnn', 'save', 'batchsize', 'lstm']

# Union - intersection
common = ['build', 'pytorch', 'google', 'tpu', 'transformer', 'loss', 'note', 'script', 'sample', 'stateoftheart', 'output', 'nlp', 'qrnn', 'policy', 'img', 'state', 'likely', 'datasets', 'parameter', 'sentence', 'answer', 'detector', 'command', 'class', 'release', 'available', '128', 'save', 'gpu', 'reward', 'system', 'compatible', 'first', 'video', 'feature', 'text', 'audio', 'place', 'accuracy', 'layer', 'set', 'wavenet', 'cuda', 'download', 'dataset', 'label', 'batch', 'token', 'per', 'pip', 'whole', 'original', 'single', 'lyric', 'runsquadpy', 'image', 'test', 'detection', 'path', 'contextual', 'speech', 'two', 'convolutional', 'environment', 'tensorflow', 'weight', 'cased', 'create', 'make', 'finetuning', 'map', 'version', 'representation', 'wordpiece', 'trained', 'usage', 'embeddings', 'folder', 'coco', 'uncased', 'action', 'dev', 'number', 'default', 'darknetexe', 'please', 'word', 'however', 'machine', 'issue', 'learning', 'landmark', 'bertbase', '110m', 'character', 'chinese', 'iteration', 'work', '2018', 'epoch', 'larger', 'multilingual', '12layer', 'yolo', 'maximum', 'one', 'cloud', 'prediction', 'example', 'large', 'bertlarge', 'running', 'memory', 'get', 'classification', 'able', 'step', 'hyperparameters', 'based', 'batchsize', 'also', 'vector', 'need', 'bert', 'like', 'pretraining', 'made', 'new', 'data', 'graph', 'time', 'tokenization', 'want', 'released', 'john', 'lstm', 'directory', 'small', 'important', 'agent', 'music', 'change', 'node', 'see', 'corpus', 'bash', 'experiment', 'reinforcement', 'sequence', 'project', 'deep', 'line', 'architecture', 'found', 'repository', 'average', '12heads', 'method', 'shell', 'train', 'valloss', 'darknet', 'mask', 'task', 'language', 'neural', 'following', 'squad', 'demo', 'length', 'export', 'input', 'recommended', 'may', '768hidden', 'mean', 'episode', 'value', 'object', 'checkpoint', 'import', 'vocabulary', 'pretrained', 'acc', 'different', 'information', '512', 'case', 'bidirectional', 'library', 'detail', 'add', 'algorithm', 'size', 'speaker', 'attention', 'update', 'embedding', 'game', 'opencv', 'question', 'window']

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

		token_list = word_tokenize(text)	# tokenize text 
		return [token for token in token_list if token not in stop_words]
		
	def remove_codeblocks(self, text):
		return re.sub('```.*?```', ' ', text)

	def remove_punctuation(self, text):
		punctuationfree="".join([i for i in text if i not in string.punctuation])
		#new_text = []
		#for word in punctuationfree:
		#	new_word=''
		#	for char in word:
		#		if char.isalnum():
		#			new_word += char
		#	if new_word != '':
		#		new_text.append(new_word)
		return punctuationfree
		#return ''.join(new_text)

	def remove_non_ascii(self, text):
		encoded_string = text.encode("ascii", "ignore")
		return encoded_string.decode()

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

	def get_keys(self,text, l):
		dict1 = {}
		for eachStr in text:
			if eachStr in dict1.keys():
				count = dict1[eachStr]
				count = count + 1
				dict1[eachStr.lower()] = count
			else: dict1[eachStr.lower()] = 1
		remekys = []
		for key in dict1:
			if dict1[key] < l or len(key) <= 2:
				remekys.append(key)
		for key in remekys:
			del dict1[key]
		return ' '.join(list(dict1.keys()))

	def keep_only_common(self, text):
		return [token for token in text if token in common]

	def run(self):
		NEWCOLNAME = TEXT
		
		#Remove codeblocks
		self.data[NEWCOLNAME]= self.data[TEXT].apply(lambda x: self.remove_codeblocks(x))
		#print('After codeblocks:\n', self.data.head())

		#Remove punctuation
		self.data[NEWCOLNAME]= self.data[NEWCOLNAME].apply(lambda x: self.remove_punctuation(x))
		#print('After punctuation:\n', self.data.head())

		#Remove non ascii
		#self.data[NEWCOLNAME]= self.data[NEWCOLNAME].apply(lambda x: self.remove_non_ascii(x))
		#This does not improve the prformance at all and it is pretty slow

		#Transfor to lowercase
		self.data[NEWCOLNAME] = self.data[NEWCOLNAME].apply(lambda x: x.lower())

		#Remove stop words
		stop_words = stopwords.words('english')
		stop_words += ['network', 'install', 'run', 'file', 'used', 'result', 'paper', 'python', 'using', 'code', 'model', 'training', 'implementation', 'use']
		self.data[NEWCOLNAME] = self.data[NEWCOLNAME].apply(lambda x : self.remove_stop_words(x, stop_words))

		#Remove links
		self.data[NEWCOLNAME] = self.data[NEWCOLNAME].apply(lambda x : self.remove_links2(x))

		#Keep only common words
		#self.data[NEWCOLNAME] = self.data[NEWCOLNAME].apply(lambda x : self.keep_only_common(x))

		#Stemming
		#porter_stemmer = PorterStemmer()
		#self.data[NEWCOLNAME]=self.data[NEWCOLNAME].apply(lambda x: self.stemming(x, porter_stemmer))

		#Lemmatizer
		wordnet_lemmatizer = WordNetLemmatizer()
		self.data[NEWCOLNAME]=self.data[NEWCOLNAME].apply(lambda x: self.lemmatizer(x, wordnet_lemmatizer))

		# Remove meaningless words
		#l = len(self.data[NEWCOLNAME])
		#self.data[NEWCOLNAME]=self.data[NEWCOLNAME].apply(lambda x: self.get_keys(x, l))
		
		#Join tokens
		self.data[NEWCOLNAME]=self.data[NEWCOLNAME].apply(lambda x: ' '.join(x))
		#print('Final: \n', self.data.head())

		
