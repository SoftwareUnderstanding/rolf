import pandas as pd
import argparse
import sys
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
	train, test = train_test_split(df, test_size=test_size, random_state=42, stratify=df['Label'])
	train.to_csv(filename.replace('.csv', '_train.csv'), sep=';', index=False)
	test.to_csv(filename.replace('.csv', '_test.csv'), sep=';', index=False)
	logthis.say('Train test set separation done')

if __name__ == "__main__":
	parser = argparse.ArgumentParser("python src/final.py", description='Perform all the steps of the moethod.')
	parser.add_argument('--preprocess_file', help='Name of .csv the file with the preprocessed data. The file will be saved in the same filename with _preprocessed at the end', required='--preprocess' in sys.argv)
	parser.add_argument('--preprocess', help='Preprocess the file before analsis', required=False, action='store_true')
	parser.add_argument('--train_test_split', help='Separate the file into train and test files.', required=False, action='store_true')
	parser.add_argument('--train_test_file', help='Name of the file to split.', required='--train_test_split' in sys.argv)
	parser.add_argument('--test_size', help='Size of the test set. (default = 0.2)', required=False, default=0.2, type=float)
	parser.add_argument('--train_models', help='Train the models.', required=False, action='store_true')
	parser.add_argument('--train_set', help='Path to train set', required='--train_models' in sys.argv)
	parser.add_argument('--test_set', help='Path to test set', required='--train_models' in sys.argv)
	parser.add_argument('--categories', help='List of categories to apply the models for. (default = all categories that are in the training set)', required=False)
	args = parser.parse_args()
	if(args.preprocess):
		preprocess_file(args.preprocess_file)
	if(args.train_test_split):
		train_test_split_file(args.train_test_file, args.test_size)
	if(args.train_models):
		print(args.train_set)
		print(args.test_set)
		train_models(args.train_set, args.test_set)
