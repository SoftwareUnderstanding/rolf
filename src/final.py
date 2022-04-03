from asyncio import run
import pandas as pd
import argparse
import sys
from evaluate_models import Evaluate

from sympy import evaluate
from Preprocessor import Preprocessor
from sklearn.model_selection import train_test_split
import logthis
from train import train_models

def preprocess_file(filename: str):
	logthis.say('Preprocessing starts')
	df = pd.read_csv(filename, sep=';')
	Preprocessor(df).run()
	df.to_csv(filename.replace('.csv', '_preprocessed.csv'), sep=';', index=False)
	logthis.say('Preprocessing done')

def train_test_split_file(filename: str, test_size = 0.2):
	logthis.say('Train test set separation starts')
	df = pd.read_csv(filename, sep=';')
	#train, test = train_test_split(df, test_size=test_size, random_state=42, stratify=df['Label'])
	train = df.drop_duplicates(subset=['Text'], keep=False)
	test = df[df.duplicated(subset=['Text'], keep=False)]
	train.to_csv(filename.replace('.csv', '_train.csv'), sep=';', index=False)
	test.to_csv(filename.replace('.csv', '_test.csv'), sep=';', index=False)
	logthis.say('Train test set separation done')

def run_experiments(models, out, test_set):
	df_test = pd.read_csv(test_set, sep=';')
	pass

def evaluate(models, test_set, categories):
	pass

if __name__ == "__main__":
	parser = argparse.ArgumentParser("python src/final.py", description='Perform all the steps of the method.')
	parser.add_argument('--preprocess_file', help='Name of .csv the file with the preprocessed data. The file will be saved in the same filename with _preprocessed at the end', required='--preprocess' in sys.argv)
	parser.add_argument('--preprocess', help='Preprocess the file before analysis', required=False, action='store_true')
	parser.add_argument('--train_test_split', help='Separate the file into train and test files.', required=False, action='store_true')
	parser.add_argument('--train_test_file', help='Name of the file to split.', required='--train_test_split' in sys.argv)
	parser.add_argument('--test_size', help='Size of the test set. (default = 0.2)', required=False, default=0.2, type=float)
	parser.add_argument('--train_models', help='Train the models.', required=False, action='store_true')
	parser.add_argument('--train_set', help='Path to train set', required='--train_models' in sys.argv or '--evaluate' in sys.argv)
	parser.add_argument('--test_set', help='Path to test set', required='--train_models' in sys.argv or '--run_experiments' in sys.argv)
	parser.add_argument('--categories', help='List of categories to apply the models for. (default = all categories that are in the training set)', required='--evaluate' in sys.argv)
	parser.add_argument('--run_experiments', help='Run experiments git the given models on the given test set', required=False, action='store_true')
	parser.add_argument('--models', help='Path to folder containing the models', required='--run_experiments' in sys.argv or '--evaluate' in sys.argv)
	parser.add_argument('--output_csv', help='Path to output csv with the results', required='--run_experiments' in sys.argv)
	parser.add_argument('--evaluate', help='Evaluate the models', required=False, action='store_true')
	args = parser.parse_args()
	if args.preprocess:
		preprocess_file(args.preprocess_file)
	if args.train_test_split:
		train_test_split_file(args.train_test_file, args.test_size)
	if args.train_models:
		train_models(args.train_set, args.test_set)
	if args.run_experiment:
		run_experiments(args.models, args.output_csv, args.test_set)
	if args.evaluate:
		evaluate(args.models, args.test_set, args.categories)
