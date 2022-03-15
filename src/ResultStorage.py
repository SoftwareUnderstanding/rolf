from pathlib import Path
import pickle
import pandas as pd
import csv

RESULTS_FILENAME = 'final_results.csv'
BESTMODEL_PATH = Path('results/models/demo')

df_results = pd.DataFrame()

class BestModel:
	def __init__(self):
		self.best_score = 0
		self.best_model = None
		self.best_pipeline = ''
	
	def addModel(self, model, score: float, pipeline: str):
		if score > self.best_score:
			self.best_model = model
			self.best_pipeline = pipeline
			self.best_score = score


class ResultStorage:
	def __init__(self, train: str, test: str, category: str):
		self.bestModel = BestModel()
		self.train = train
		self.test = test
		self.category = category

	def processResult(self, results, model):
		self.bestModel.addModel(model, results['f1-score_overall'].values[0], results['Pipeline'].values[0])
		global df_results
		df_results = df_results.append(results)
		self.writeResults('results.csv', results)

	def writeResults(self, results_filename : str, df_results : pd.DataFrame):
		with open('results/' + results_filename, 'a+') as csvfile:
			csvWriter = csv.writer(csvfile, delimiter=';')
			csvWriter.writerow([df_results['PipelineID'].iloc[-1],
								df_results['Pipeline'].iloc[-1],
								df_results['test_acc_mean'].iloc[-1],
								df_results['test_prec_mean'].iloc[-1],
								df_results['test_recall_mean'].iloc[-1],
								df_results['test_f1-score_mean'].iloc[-1],
								df_results['acc_overall'].iloc[-1],
								df_results['prec_overall'].iloc[-1],
								df_results['recall_overall'].iloc[-1],
								df_results['f1-score_overall'].iloc[-1],
								self.train,
								self.test])
	def dumpResults(self, filename: str = None):
		if filename is None:
			df_results.to_csv(RESULTS_FILENAME, sep=';')
		else:
			df_results.to_csv(filename, sep=';')

	def dumpBestModel(self, filename: str = None):
		if filename is None:
			BESTMODEL_PATH.mkdir(parents=True, exist_ok=True)
			pickle.dump(self.bestModel.best_model, open(BESTMODEL_PATH / (f'{self.category.replace(" ", "_").lower()}_{self.bestModel.best_pipeline}_.sav'), 'wb'))
		else:
			pickle.dump(self.bestModel.best_model, open(filename, 'wb'))
