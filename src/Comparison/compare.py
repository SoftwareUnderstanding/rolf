import json
import sys
import os

sys.path.append(os.path.abspath(os.getcwd()) + '/src')
from Evaluation import evaluation

evaluator = evaluation.Evaluator('data/csoc_output_transformed.csv', prediction_fieldname='Csoc_predictions', transformer=lambda x: evaluation.csoc_transform_predictions(x, {'sequential programming': 'sequential', 'graph theory': 'graphs', 'digital audio': 'audio'}))
json.dump(open('results/compare_results/csoc_onetoone_mapping.json', 'w'), evaluator.evaluate(), indent=4)

evaluator = evaluation.Evaluator('data/demo1_predictions/demo1.csv')
json.dump(open('results/compare_results/aimmx.json', 'w'), evaluator.evaluate(), indent=4)

evaluator = evaluation.Evaluator('data/aimmx_output_transformed.csv')
json.dump(open('results/compare_results/demo1.json', 'w'), evaluator.evaluate(), indent=4)
