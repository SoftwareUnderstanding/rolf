import argparse
import csv
from pathlib import Path
import pickle
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from bert.tokenization import bert_tokenization as tokenization

class Predictor:
	"""
	This class is used to run predictions with given models on given datasets and save the results. Also collects all the labels.

	Methods
	----------
	predict: Used to load the given models, run the predictions and save the results.
	saveData: Used to save the prediction data.

	Attributes
	----------
	models_path: (Path) Path to the folder containing the models used to predict.
	data_path: (str) Path to the csv file containing the samples.
	out_data: (Dict) Used to store the predictions.
	out_path: (Path) Path to the output file. Defaults to {models_path}/predictions/{models_folder}.csv
	"""
	def __init__(self, models_path: str, data_path: str, out_path: str = None, bert: bool = False):
		"""
		Params
		--------
		models_path: (Path) Path to the folder containing the models used to predict.
		data_path: (str) Path to the csv file containing the samples.
		out_path: (Path) Path to the output file. Defaults to {models_path}/predictions/{models_folder}.csv
		"""
		self.models_path = Path(models_path)
		self.data_path = data_path
		self.bert = bert
		if out_path is not None:
			self.out_path = Path(out_path)
		else:
			model_path_path = Path(models_path)
			model_path_path = model_path_path / f'predictions/{model_path_path.name}.csv'
			model_path_path.parent.mkdir(parents=True, exist_ok=True)
			self.out_path = model_path_path
		self.out_path.parent.mkdir(parents=True, exist_ok=True)
		self.out_data = {}

	def __loadModels(self) -> None:
		"""
		Loads and stores all the models from the given folder.
		"""
		self.__models = []
		self.model_labels = []
		for model_file in self.models_path.iterdir():
			if self.bert:
				self.__models.append(tf.keras.models.load_model(model_file.absolute().as_posix(), custom_objects={'KerasLayer': hub.KerasLayer}))
			else:
				with open(model_file, 'rb') as f:
					self.model_labels.append(model_file.name.split('.')[0].replace('_', ' '))
					self.__models.append(pickle.load(f))

	def saveData(self) -> None:
		"""
		Saves the predictions into {self.out_path} csv file with ['Labels', 'Repo', 'Predictions'] columns.
		"""
		with open(self.out_path, 'w') as f:
			writer = csv.DictWriter(f, delimiter=';', fieldnames=['Labels', 'Repo', 'Predictions', 'Probabilities'])
			writer.writeheader()
			for key, val in self.out_data.items():
				writer.writerow({
					'Labels': ','.join(val['Labels']),
					'Repo': key,
					'Predictions': ','.join(val['Predictions']),
					'Probabilities': ','.join(val['Probabilities']),
				})

	def encoding_the_text(self, texts, tokenizer, max_len=300):
		all_tokens = []
		all_masks = []
		all_segments = []

		for text in texts:
			text = tokenizer.tokenize(text)

			text = text[:max_len-2]
			input_sequence = ["[CLS]"] + text + ["[SEP]"]
			pad_len = max_len-len(input_sequence)

			tokens = tokenizer.convert_tokens_to_ids(input_sequence) + [0] * pad_len
			pad_masks = [1] * len(input_sequence) + [0] * pad_len
			segment_ids = [0] * max_len

			all_tokens.append(tokens)
			all_masks.append(pad_masks)
			all_segments.append(segment_ids)
		return np.array(all_tokens), np.array(all_masks), np.array(all_segments)

	def getBertTokenizer(self):
		m_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2'
		bert_layer = hub.KerasLayer(m_url, trainable=True)
		tf.gfile = tf.io.gfile
		vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
		do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
		return tokenization.FullTokenizer(vocab_file, do_lower_case)

	def getLabelsFromPrediction(self, prediction) -> set[str]:
		categories_alphabetical = ['Audio', 'Computer Vision', 'Graphs', 'Natural Language Processing', 'Reinforcement Learning', 'Sequential']
		print(prediction)
		above_thresholds_indices = np.argwhere(prediction > 0.5)
		res = []
		if above_thresholds_indices:
			for ind in above_thresholds_indices:
				res.append(categories_alphabetical[ind[0]])
		return set(res)

	def predict(self):
		"""
		Loads and stores all the models from the given folder.\n
		Runs and collects the predictions on the given samples.\n
		Also collects and merges all the labels for the repositories.\n
		Saves the predictions by calling saveData().
		"""
		self.__loadModels()
		self.out_data = {}


		if self.bert:
			tokenizer = self.getBertTokenizer()

			with open(self.data_path) as f:
				reader = csv.DictReader(f, delimiter=';')
				text_data = [row for row in reader]
				texts = [row['Text'] for row in text_data]
			text = self.encoding_the_text(texts, tokenizer)
			print(len(texts), len(text))
			predictions = [model.predict(text) for model in self.__models]
			for i, row in enumerate(text_data):
				repo = row['Repo']
				if repo in self.out_data:
					self.out_data[repo]['Labels'].add(row['Label'])
				else:
					self.out_data[repo] = {'Labels' : set([row['Label']]), 'Predictions': set()}
					for prediction in predictions:
						labels = self.getLabelsFromPrediction(prediction[i])
						self.out_data[repo]['Probabilities'] = prediction[i]
						print(labels)
						self.out_data[repo]['Predictions'].update(labels)
		else:
			with open(self.data_path) as f:
				reader = csv.DictReader(f, delimiter=';')
				for row in reader:
					repo = row['Repo']
					if repo in self.out_data:
						self.out_data[repo]['Labels'].add(row['Label'])
					else:
						self.out_data[repo] = {'Labels' : set([row['Label']]), 'Predictions': set(), 'Probabilities': []}
						for model_ind, model in enumerate(self.__models):
							pred = model.predict([row['Text']])
							pred_proba = model.predict_proba([row['Text']])
							model_label = self.model_labels[model_ind]
							#print(f'{model_label=}')
							#print(pred, pred_proba)
							if 'Other' > model_label:
								self.out_data[repo]['Probabilities'].append(str(pred_proba[0][0]))
							else:
								self.out_data[repo]['Probabilities'].append(str(pred_proba[0][1]))
							if pred != 'Other':
								self.out_data[repo]['Predictions'].add(pred[0])
		self.saveData()

if __name__ == '__main__':
	parser_predict = argparse.ArgumentParser('python src/Evaluation/prediction.py', description='Predict with the given models.')
	parser_predict.add_argument('--inputfolder', required=True, help='Path of folder with the models.')
	parser_predict.add_argument('--test_set', required=True, help='Name of the csv file containing the test set.')
	parser_predict.add_argument('--outfile', required=True, help='Path to output csv file with the results.')
	parser_predict.add_argument('--bert', action=argparse.BooleanOptionalAction)

	args = parser_predict.parse_args()

	predictor = Predictor(args.inputfolder, args.test_set, args.outfile, args.bert).predict()

