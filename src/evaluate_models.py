import pandas as pd
import pathlib
import pickle

class Evaluate:
	def __init__(self, model_path, test_set, category):
		self.model = pickle.load(open(model_path, 'rb'))
		self.test_set = pd.read_csv(test_set, sep=';')
		self.category = category
	
	def predict(self):
		self.model.predict_proba()
