from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from tqdm import tqdm
import fasttext
import pickle
import pandas as pd
import numpy as np


def createWordEmbedding(df : pd.DataFrame, textcolname : str):
	# create a tokenizer 
	token = Tokenizer(oov_token='<OOV>')
	token.fit_on_texts(df[textcolname])
	word_index = token.word_index
	
	# create token-embedding mapping
	pretrained = fasttext.FastText.load_model('src_new/crawl-300d-2M-subword.bin')
	embedding_matrix = np.zeros((len(word_index) + 1, 300))
	words = []
	for word, i in tqdm(word_index.items()):
		embedding_vector = pretrained.get_word_vector(word) #embeddings_index.get(word)
		words.append(word)
		if embedding_vector is not None:
			embedding_matrix[i] = embedding_vector

	filename = 'word_embedding.sav'
	pickle.dump(token, open( 'results/models/' + filename, 'wb'))

	return word_index, embedding_matrix