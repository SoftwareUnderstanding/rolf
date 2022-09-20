import argparse
import csv
import json
import os
import sys
from typing import Any, Callable, Dict, Iterable, Set
import logthis

sys.path.append(os.path.abspath(os.getcwd()) + '/src')

from util.utils import getCategories, BASE_CATEGORIES

def lower_transform(predictions: Iterable[str]) -> Set[str]:
	"""
	Transforms the items of the given Iterable with str.lower() function.\n
	Used as a default transform function.

	Params
	---------
	predictions: (Iterable[str]) Collection of texts to transform.

	Return
	---------
	(Set[str]): Set of transformed items.
	"""
	return {pred.lower() for pred in predictions}

def csoc_transform_predictions(predictions: Iterable[str], transform_dict: Dict[str, str]) -> Set[str]:
	"""
	Transforms the items of the given Iterable with str.lower() function and given mapping after.\n
	Used to transform predictions of CSOC.

	Params
	---------
	predictions: (Iterable[str]) Collection of texts to transform.
	transform_dict: (Dict[str]) Key-value pairs used to map CSOC predictions into known categories.

	Return
	---------
	(Set[str]): Set of transformed items.
	"""
	ret = set()
	logthis.say('original: '+ str(predictions))
	for prediction in predictions:
		pred = prediction.lower()
		if pred in transform_dict:
			ret.add(transform_dict[pred])
		else:
			ret.add(pred)
	logthis.say('mapped: '+ str(ret))
	return ret

class Evaluator:

	"""
	This class is used to evaluate the predictions and save the collected statistics.

	Methods
	----------
	resetStats: Initializes or resets the collected stats.
	evaluate: Runs the evaluation and collects stats.
	dumpStats: Dumps the collected stats in JSON format.

	Attributes
	---------
	inputfile: (str) Used to store the input file with predictions.
	categories: (Iterable[str]) Collection of categories to consider when collecting stats.
	transformer: (Callable[[Iterable[str]], Set[str]]) Used to store the transform function to transform predictions into categories.
	stat_fields: (List[str]) Used to store the fields used in statistics.
	prediction_fieldname: (str) Name of the column where the prediction values are stored in the csv file.
	"""

	def __init__(self, 
				inputfile: str, 
				categories: Iterable[str] = None, 
				transformer: Callable[[Iterable[str]], Set[str]] = lower_transform, 
				prediction_fieldname: str = 'Predictions'):
		"""
		Params
		---------
		inputfile: (str) Used to store the input file with predictions.
		categories: (Iterable[str]) Collection of categories to consider when collecting stats.
					Defaults to {'natural language processing', 'general', 'sequential', 'computer vision', 'reinforcement learning', 'graphs', 'audio'}.
		transformer: (Callable[[Iterable[str]], Set[str]]) Used to store the transform function to transform predictions into categories.
					Defaults to lower_transform, which performs str.lower() on predictions.
		prediction_fieldname: (str) Name of the column where the prediction values are stored in the csv file.
					Defaults to 'Predictions'.
		"""
		self.inputfile = inputfile
		self.categories = set([cat.lower() for cat in categories]) if categories is not None else {'natural language processing', 'general', 'sequential', 'computer vision', 'reinforcement learning', 'graphs', 'audio'}
		self.transformer = transformer		
		self.stat_fields = ['tp', 'tn', 'fp', 'fn', 'support']
		self.resetStats()
		self.prediction_fieldname = prediction_fieldname

	def resetStats(self) -> None:
		"""
		Initializes or resets the collected stats.
		"""
		self.stats = {
						'overall' : {key: 0 for key in self.stat_fields},
						'overall_presentonly' : {key: 0 for key in self.stat_fields},
					}
		for category in self.categories:
			self.stats[category] = {key: 0 for key in self.stat_fields}

	def evaluate(self) -> Dict[str, Dict[str, Any]]:
		"""
		Runs the evaluation and collects stats.

		Return
		--------
		(Dict[str, Dict[str, Any]]) Returns the collected statistics. 
		"""
		with open(self.inputfile) as f:
			self.reader = csv.DictReader(f, delimiter=';')
			for row in self.reader:
				self.stats['overall']['support'] += 1
				predictions = self.transformer(row[self.prediction_fieldname].split(','))
				labels = lower_transform(row['Labels'].split(','))
				for category in self.categories:
					self.stats[category]['support'] += 1
					if category in predictions:
						if category in labels:
							self.stats['overall']['tp'] += 1
							self.stats[category]['tp'] += 1
						else:
							self.stats['overall']['fp'] += 1
							self.stats[category]['fp'] += 1
					else:
						if category in labels:
							self.stats['overall']['fn'] += 1
							self.stats[category]['fn'] += 1
						else:
							self.stats['overall']['tn'] += 1
							self.stats[category]['tn'] += 1

		for category in self.categories:
			if self.stats[category]['tp'] != 0 or self.stats[category]['fp'] != 0:
				for key in self.stat_fields:
					self.stats['overall_presentonly'][key] += self.stats[category][key]
				self.stats['overall_presentonly']['support'] += self.stats['overall']['support']

		for key, val in self.stats.items():
			self.stats[key]['precision'] = f"{val['tp']/(1 + val['tp'] + val['fp']):.2f}"
			self.stats[key]['recall'] = f"{val['tp']/(1 + val['tp'] + val['fn']):.2f}"
			self.stats[key]['sample_num'] = val['tp'] + val['fn']

		return self.stats.copy()

	def dumpStats(self, outfile: str):
		"""
		Dumps the collected stats in JSON format.

		Params
		---------
		outfile: (str) Path to the output file.
		"""
		with open(outfile, 'w') as f:
			json.dump(self.stats, f, indent=4)
		
if __name__ == '__main__':
	parser_evaluate = argparse.ArgumentParser('python src/Evaluation/evaluate.py', description='Evaluate the predictions.')
	parser_evaluate.add_argument('--inputfile', required=True, help='Path of the csv file with the predictions.')
	parser_evaluate.add_argument('--outfile', required=True, help='Path of the json file to write scores.')
	parser_evaluate_categories = parser_evaluate.add_mutually_exclusive_group(required=False)
	parser_evaluate_categories.add_argument('--all_categories', nargs="+", help=f'List of all categories used. Use only if you want not the basic categories. {BASE_CATEGORIES=}')
	parser_evaluate_categories.add_argument('--additional_categories', nargs="+", help=f'List of categories adding to basic categories. {BASE_CATEGORIES=}')
	args = parser_evaluate.parse_args()
	
	categories = getCategories(BASE_CATEGORIES, args.all_categories, args.additional_categories)
	logthis.say(f"{categories=}")
	evaluator = Evaluator(args.inputfile, set(categories))
	evaluator.evaluate()
	evaluator.dumpStats(args.outfile)
