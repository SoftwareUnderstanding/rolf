import argparse
from typing import Iterable, List
import logthis
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import numpy as np
import inflect
import contractions
from bs4 import BeautifulSoup
import re, unicodedata
from nltk import word_tokenize
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from nltk.stem.api import StemmerI

# Words that only apear in one category
#common = ['audio', 'pip', 'sample', 'music', 'wavenet', 'lyric', 'valloss', 'speaker', 'speech', 'cuda', 'folder', 'average', 'opencv', 'darknet', 'coco', 'demo', 'create', 'iteration', 'video', 'window', 'graph', 'acc', 'convolutional', 'feature', 'node', 'embedding', 'datasets', '110m', '12heads', 'available', 'made', 'released', 'original', 'found', 'experiment', 'larger', 'add', 'representation', 'update', 'bidirectional', 'answer', 'large', 'information', 'finetuning', 'transformer', 'detail', 'usage', 'dev', '12layer', 'based', 'method', '128', 'chinese', 'nlp', 'small', 'library', 'google', 'token', 'length', 'corpus', 'stateoftheart', 'character', 'may', 'however', 'likely', 'cased', '2018', 'release', 'contextual', 'export', 'tokenization', 'classification', 'multilingual', '768hidden', 'vocabulary', 'bertlarge', 'per', 'able', 'hyperparameters', 'accuracy', 'whole', 'running', 'bertbase', 'import', 'compatible', 'two', 'recommended', 'system', 'mask', 'architecture', 'work', 'tpu', '512', 'wordpiece', 'runsquadpy', 'attention', 'important', 'place', 'maximum', 'john', 'shell', 'uncased', 'case', 'vector', 'mean', 'issue', 'question', 'environment', 'agent', 'state', 'reward', 'action', 'episode', 'landmark', 'game', 'reinforcement', 'policy', 'qrnn', 'save', 'batchsize', 'lstm']

# Union - intersection
#common = ['build', 'pytorch', 'google', 'tpu', 'transformer', 'loss', 'note', 'script', 'sample', 'stateoftheart', 'output', 'nlp', 'qrnn', 'policy', 'img', 'state', 'likely', 'datasets', 'parameter', 'sentence', 'answer', 'detector', 'command', 'class', 'release', 'available', '128', 'save', 'gpu', 'reward', 'system', 'compatible', 'first', 'video', 'feature', 'text', 'audio', 'place', 'accuracy', 'layer', 'set', 'wavenet', 'cuda', 'download', 'dataset', 'label', 'batch', 'token', 'per', 'pip', 'whole', 'original', 'single', 'lyric', 'runsquadpy', 'image', 'test', 'detection', 'path', 'contextual', 'speech', 'two', 'convolutional', 'environment', 'tensorflow', 'weight', 'cased', 'create', 'make', 'finetuning', 'map', 'version', 'representation', 'wordpiece', 'trained', 'usage', 'embeddings', 'folder', 'coco', 'uncased', 'action', 'dev', 'number', 'default', 'darknetexe', 'please', 'word', 'however', 'machine', 'issue', 'learning', 'landmark', 'bertbase', '110m', 'character', 'chinese', 'iteration', 'work', '2018', 'epoch', 'larger', 'multilingual', '12layer', 'yolo', 'maximum', 'one', 'cloud', 'prediction', 'example', 'large', 'bertlarge', 'running', 'memory', 'get', 'classification', 'able', 'step', 'hyperparameters', 'based', 'batchsize', 'also', 'vector', 'need', 'bert', 'like', 'pretraining', 'made', 'new', 'data', 'graph', 'time', 'tokenization', 'want', 'released', 'john', 'lstm', 'directory', 'small', 'important', 'agent', 'music', 'change', 'node', 'see', 'corpus', 'bash', 'experiment', 'reinforcement', 'sequence', 'project', 'deep', 'line', 'architecture', 'found', 'repository', 'average', '12heads', 'method', 'shell', 'train', 'valloss', 'darknet', 'mask', 'task', 'language', 'neural', 'following', 'squad', 'demo', 'length', 'export', 'input', 'recommended', 'may', '768hidden', 'mean', 'episode', 'value', 'object', 'checkpoint', 'import', 'vocabulary', 'pretrained', 'acc', 'different', 'information', '512', 'case', 'bidirectional', 'library', 'detail', 'add', 'algorithm', 'size', 'speaker', 'attention', 'update', 'embedding', 'game', 'opencv', 'question', 'window']

# Union
#common = ['network', 'install', 'run', 'file', 'used', 'result', 'paper', 'python', 'using', 'code', 'model', 'training', 'implementation', 'use']

class Preprocessor:
	"""
	A class to perform preprocessing on given DataFrame.

	Attributes
	----------
	data : pd.DataFrame
		The pandas DataFrame containing the data.
	TEXT : str
		Column name containing the data to preprocess (Default : 'Text')

	Methods
	-------
	run()
		Performs all the preprocessing methods on the data in-place.
	"""

	def __init__(self, data : pd.DataFrame, TEXT: str = 'Text') -> None:
		"""
		Parameters
		----------
		data : pd.DataFrame
			The DataFrame containing the data.
		TEXT : str
			Column name containing the data to preprocess (Default : 'Text')
		"""

		self.data = data
		self.TEXT = TEXT

	def denoise_text(self, text: str) -> str:
		"""
			Strip html if any. For ex. removing <html>, <p> tags.\n
			Replace contractions in the text. For ex. didn't -> did not.\n
			Remove '404: Not Found' from text.
		"""
		soup = BeautifulSoup(text, "html.parser")
		text = soup.get_text()
		text = contractions.fix(text)
		text.replace("""404: Not Found""", '')
		return text

	def remove_stop_words(self, text : str) -> List[str]:
		"""
		Removes stop words (meaningless common words) from text. 
		"""
		stop_words = stopwords.words('english')
		stop_words += ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'hundred', 'thousand', 'and']
		stop_words += ['network', 'install', 'run', 'file', 'use', 'result', 'paper', 'python', 'using', 'code', 'model', 'train', 'implementation', 'use']
		stop_words += ['data', 'dataset', 'example', 'build', 'learn', 'download', 'obj']
		#stop_words += ['html', 'one', 'two', 'three', 'etc', 'x64', 'instead', 'repository', 'please', 'also', 'project', 'google', 'following', 'get', 'see']
		#stop_words += ['likely', 'may', 'want', '110m', 'like', 'made', 'example', 'able', 'first', 'however', 'need', 'make', 'new', 'reference']
		return [word for word in text if not word in stop_words]
		
	def remove_codeblocks(self, text: str) -> str:
		"""Removes code blocks from text (```.*?``` pattern.)"""
		return re.sub('```.*?```', ' ', text)

	def remove_punctuation(self, text: str) -> str:
		"""Removes punctuations from text (non-word and not whitespace characters, plus underline)."""
		res = re.sub(r'[^\w\s]|\_', ' ', text)
		return res

	def remove_non_ascii(self, words: Iterable[str]) -> List[str]:
		"""Removes non-ASCII characters from list of tokenized words"""
		new_words = []
		for word in words:
			new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
			new_words.append(new_word)
		return new_words

	def replace_numbers(self, words: Iterable[str]) -> List[str]:
		"""Replace all integer occurrences in list of tokenized words with textual representation."""
		p = inflect.engine()
		new_words = []
		for word in words:
			if word.isdigit():
				new_word = p.number_to_words(word)
				new_words += new_word.split(' ')
			else:
				new_words.append(word)
		return new_words

	def stemming(self, text: Iterable[str], porter_stemmer: StemmerI) -> List[str]:
		"""Stem words in list of tokenized words with given stemmer"""
		stem_text = [porter_stemmer.stem(word) for word in text]
		return stem_text

	def stem_words(self, words: Iterable[str]) -> List[str]:
		"""Stem words in list of tokenized words"""
		stemmer = LancasterStemmer()
		stems = []
		for word in words:
			stem = stemmer.stem(word)
			stems.append(stem)
		return stems
	
	def lemmatize_verbs(self, words: Iterable[str]) -> List[str]:
		"""Lemmatize verbs in list of tokenized words"""
		lemmatizer = WordNetLemmatizer()
		lemmas = []
		for word in words:
			lemma = lemmatizer.lemmatize(word, pos='v')
			lemmas.append(lemma)
		return lemmas

	def lemmatize_nouns(self, words: Iterable[str]) -> List[str]:
		"""Lemmatize nouns in list of tokenized words"""
		lemmatizer = WordNetLemmatizer()
		lemmas = []
		for word in words:
			lemma = lemmatizer.lemmatize(word, pos='n')
			lemmas.append(lemma)
		return lemmas

	def lemmatize_adjectives(self, words: Iterable[str]) -> List[str]:
		"""Lemmatize adjectives in list of tokenized words"""
		lemmatizer = WordNetLemmatizer()
		lemmas = []
		for word in words:
			lemma = lemmatizer.lemmatize(word, pos='a')
			lemmas.append(lemma)
		return lemmas

	def remove_short_and_number_words(self, text: Iterable[str]) -> List[str]:
		"""Removes numbers and one or two character long words."""
		res = [word for word in text if word.isdigit() == False and len(word) > 2]
		return res

	def remove_links(self, text: str) -> str:
		"""Remove links from the text by using regex."""
		regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
		return (re.sub(regex, '', text))

	def remove_links2(self, text: str) -> str:
		"""Remove links from the text by checking 'http' prefix."""
		return ' '.join([token for token in text.split(' ') if 'http' not in token])

	#def keep_only_common(self, text: Iterable[str]) -> List[str]:
	#	"""Remove preset common words from text."""
	#	return [token for token in text if token in common]

	def run(self):
		"""
		Main method of the class, performing all the preprocessing methods on the data in-place. Dropping empty rows after.
		"""

		#List preprocessing methods
		pipeline = {
			'remove codeblocks': lambda x: self.remove_codeblocks(x),
			'remove links': lambda x : self.remove_links2(x),
			'remove tags': lambda x : self.denoise_text(x),
			'remove punctuations': lambda x: self.remove_punctuation(x),
			'transform to lowercase': lambda x: x.lower(),
			#######'replace numbers': lambda x : self.replace_numbers(word_tokenize(x)),
			'remove non-ascii characters': lambda x : self.remove_non_ascii(word_tokenize(x)),
			'lemmatize verbs': lambda x : self.lemmatize_verbs(x),
			'lemmatize nouns': lambda x : self.lemmatize_nouns(x),
			'lemmatize adjectives': lambda x : self.lemmatize_adjectives(x),
			'remove stop_words': lambda x : self.remove_stop_words(x),
			'remove tokens only containing numbers or two char': lambda x : self.remove_short_and_number_words(x),
			#######'keep only common words': lambda x : self.keep_only_common(x),
			#######'stemming': lambda x: self.stemming(x, PorterStemmer()),
			'join tokens': lambda x: ' '.join(x),
		}

		#Perform preprocessors
		for key, val in pipeline.items():
			logthis.say(f'Preprocessing: Process name: "{key}".')
			self.data[self.TEXT] = self.data[self.TEXT].apply(val)

		#Drop empty rows
		logthis.say("Preprocessing: drop empty rows.")
		self.data.drop(self.data[self.data[self.TEXT] == np.nan].index, inplace=True)
		self.data.drop(self.data[self.data[self.TEXT] == ''].index, inplace=True)


def preprocess_file(filename: str) -> None:
	"""
	Reads the data from the given csv file with ";" separator, runs preprocessing methods on the data, 
	and saves the preprocessed data next to the original file with "_preprocessed.csv" suffix.

	Params
	---------		
	filename: (str) Path to the input csv file.
	"""
	logthis.say('Preprocessing starts.')
	df = pd.read_csv(filename, sep=';')
	Preprocessor(df).run()
	df.to_csv(filename.replace('.csv', '_preprocessed.csv'), sep=';', index=False)
	logthis.say('Preprocessing done.')

if __name__ == "__main__":
	parser_preprocess = argparse.ArgumentParser('python src/preprocessing.py', description="Preprocess given csv data file.")
	parser_preprocess.add_argument('--preprocess_file', required=True, help='Name of .csv the file with the preprocessed data. The file will be saved in the same filename with "_preprocessed" suffix.')
	
	args = parser_preprocess.parse_args()

	preprocess_file(args.preprocess_file)
