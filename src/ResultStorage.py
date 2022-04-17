from pathlib import Path
import pickle
import pandas as pd
import csv

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
	def __init__(self, train: str, test: str, category: str, evaluation_metric: str = "test_f1-score_mean"):
		self.bestModel = BestModel()
		self.train = train
		self.test = test
		self.category = category
		self.evaluation_metric = evaluation_metric
		self.df_results = pd.DataFrame()

	def processResult(self, results: pd.DataFrame, model) -> None:
		self.bestModel.addModel(model, results[self.evaluation_metric].values[0], results['Pipeline'].values[0])
		self.df_results = pd.concat([self.df_results, results])
		#self.writeResults('results.csv', self.results)

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

	def dumpResults(self, filename: str) -> None:
		Path(filename).parent.mkdir(parents=True, exist_ok=True)
		self.df_results.to_csv(filename, sep=';', index=False)

	def dumpBestModel(self, folder_name: str) -> None:
		Path(folder_name).mkdir(parents=True, exist_ok=True)
		with open(f'{folder_name}/{self.category.replace(" ", "_").lower()}_{self.bestModel.best_pipeline}.sav', 'wb') as f:
			pickle.dump(self.bestModel.best_model, f)
