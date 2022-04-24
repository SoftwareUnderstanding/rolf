import argparse
import csv
from pathlib import Path
import pickle

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
	def __init__(self, models_path: str, data_path: str, out_path: str = None):
		"""
		Params
		--------
		models_path: (Path) Path to the folder containing the models used to predict.
		data_path: (str) Path to the csv file containing the samples.
		out_path: (Path) Path to the output file. Defaults to {models_path}/predictions/{models_folder}.csv
		"""
		self.models_path = Path(models_path)
		self.data_path = data_path
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
		for model_file in self.models_path.iterdir():
			with open(model_file, 'rb') as f:
				self.__models.append(pickle.load(f))

	def saveData(self) -> None:
		"""
		Saves the predictions into {self.out_path} csv file with ['Labels', 'Repo', 'Predictions'] columns.
		"""
		with open(self.out_path, 'w') as f:
			writer = csv.DictWriter(f, delimiter=';', fieldnames=['Labels', 'Repo', 'Predictions'])
			writer.writeheader()
			for key, val in self.out_data.items():
				writer.writerow({
					'Labels': ','.join(val['Labels']),
					'Repo': key,
					'Predictions': ','.join(val['Predictions'])
				})

	def predict(self):
		"""
		Loads and stores all the models from the given folder.\n
		Runs and collects the predictions on the given samples.\n
		Also collects and merges all the labels for the repositories.\n
		Saves the predictions by calling saveData().
		"""
		self.__loadModels()
		self.out_data = {}
		with open(self.data_path) as f:
			reader = csv.DictReader(f, delimiter=';')
			for row in reader:
				repo = row['Repo']
				if repo in self.out_data:
					self.out_data[repo]['Labels'].add(row['Label'])
				else:
					self.out_data[repo] = {'Labels' : set([row['Label']]), 'Predictions': set()}
					for model in self.__models:
						pred = model.predict([row['Text']])[0]
						if pred != 'Other':
							self.out_data[repo]['Predictions'].add(pred)
		self.saveData()

if __name__ == '__main__':
	parser_predict = argparse.ArgumentParser('python src/Evaluation/prediction.py', description='Predict with the given models.')
	parser_predict.add_argument('--inputfolder', required=True, help='Path of folder with the models.')
	parser_predict.add_argument('--test_set', required=True, help='Name of the csv file containing the test set.')
	parser_predict.add_argument('--outfile', required=True, help='Path to output csv file with the results.')
	
	args = parser_predict.parse_args()

	predictor = Predictor(args.inputfolder, args.test_set, args.outfile).predict()

