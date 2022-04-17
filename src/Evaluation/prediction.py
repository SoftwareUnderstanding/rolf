import csv
from pathlib import Path
import pickle

class Predictor:
	
	def __init__(self, models_path: str, data_path: str, out_path: str = None):
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

	def loadModels(self):
		self.__models = []
		for model_file in self.models_path.iterdir():
			with open(model_file, 'rb') as f:
				self.__models.append(pickle.load(f))

	def predict(self):
		self.loadModels()
		out_data = {}
		with open(self.data_path) as f:
			reader = csv.DictReader(f, delimiter=';')
			for row in reader:
				repo = row['Repo']
				if repo in out_data:
					out_data[repo]['Labels'].add(row['Label'])
				else:
					out_data[repo] = {'Labels' : set([row['Label']]), 'Predictions': set()}
					for model in self.__models:
						pred = model.predict([row['Text']])[0]
						if pred != 'Other':
							out_data[repo]['Predictions'].add(pred)
		
		with open(self.out_path, 'w') as f:
			writer = csv.DictWriter(f, delimiter=';', fieldnames=['Labels', 'Repo', 'Predictions'])
			writer.writeheader()
			for key, val in out_data.items():
				writer.writerow({
					'Labels': ','.join(val['Labels']),
					'Repo': key,
					'Predictions': ','.join(val['Predictions'])
				})

if __name__ == '__main__':
	models_path = Path('results/models/demo1')
	predictor = Predictor(models_path, 'data/readme_new_preprocessed_test.csv', f'data/demo1_predictions/{models_path.name}.csv')
	predictor.predict()