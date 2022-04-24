from pathlib import Path
import pickle
import pandas as pd

class ResultStorage:
	"""
	This class can be used to store results of training methods. The class stores all the training results and the best model 
	according to the given evaluation metric.

	Methods
	-----------
	processResult: This method is used to feed the object with the results.
	dumpResults: Writes all the training results to the given file in csv format.
	dumpBestModel: Saves the best model object with pickle.
	"""

	class BestModel:
		"""
		This inner class is used to store the best model so far.

		Methods
		----------
		addModel: Used to add the new model for evaluation.

		Attributes
		----------
		best_score: (float) Used to store the best score so far.
		best_model: (object) Used to store the model with the best score so far.
		best_pipeline: (str) Used to store the name of the pipeline with the best score so far.
		"""
		def __init__(self):
			self.best_score: float = 0
			self.best_model = None
			self.best_pipeline = ''
		
		def addModel(self, model, score: float, pipeline: str) -> None:
			"""
			Used to add the new model for evaluation.

			Params
			---------
			model: (object) The trained model.
			score: (float) The score value used for evaluation.
			pipeline: (str) Name of the pipeline (as you would like to name it).
			"""
			if score > self.best_score:
				self.best_model = model
				self.best_pipeline = pipeline
				self.best_score = score

	def __init__(self, category: str, evaluation_metric: str = "test_f1-score_mean"):
		"""
		Params
		----------
		category: (str) Name of the category the training is running on (used in filenames).
		evaluation_metric: (str) The key used to evaluate the best model from training results.
		"""
		self.bestModel = ResultStorage.BestModel()
		self.category = category
		self.evaluation_metric = evaluation_metric
		self.df_results = pd.DataFrame()

	def processResult(self, results: pd.DataFrame, model) -> None:
		"""
		This method is used to feed the object with the results.

		Params
		----------
		results: (pandas.DataFrame) Result data of the training.
		model: (object) The result model of the training.
		"""
		self.bestModel.addModel(model, results[self.evaluation_metric].values[0], results['Pipeline'].values[0])
		self.df_results = pd.concat([self.df_results, results])

	def dumpResults(self, filename: str) -> None:
		"""
		Writes all the training results to the given file in csv format.

		Params:
		---------
		filename: (str) Path to the file where the data will be dumped.
		"""
		Path(filename).parent.mkdir(parents=True, exist_ok=True)
		self.df_results.to_csv(filename, mode='a', sep=';', index=False)

	def dumpBestModel(self, folder_name: str) -> None:
		"""
		Saves the best model object with pickle.

		Params:
		---------
		filename: (str) Path to the folder where the object will be saved.
		"""
		Path(folder_name).mkdir(parents=True, exist_ok=True)
		with open(f'{folder_name}/{self.category.replace(" ", "_").lower()}_{self.bestModel.best_pipeline}.sav', 'wb') as f:
			pickle.dump(self.bestModel.best_model, f)
