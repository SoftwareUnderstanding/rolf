import json
import sys
from typing import List
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
import logthis

from Preprocessor import Preprocessor
from Evaluation.evaluation import Evaluator
from Evaluation.prediction import Predictor
import collectreadmes

BASE_CATEGORIES = ["Natural Language Processing", "Computer Vision", "Sequential", "Audio", "Graphs", "Reinforcement Learning"]

def preprocess_file(filename: str):
	logthis.say('Preprocessing starts.')
	df = pd.read_csv(filename, sep=';')
	Preprocessor(df).run()
	df.to_csv(filename.replace('.csv', '_preprocessed.csv'), sep=';', index=False)
	logthis.say('Preprocessing done.')

def train_test_split_file(filename: str, test_size: float = 0.2, new_category: bool = True) -> None:
	logthis.say('Train test set separation starts.')
	df: pd.DataFrame = pd.read_csv(filename, sep=';')
	if new_category:
		train, test = train_test_split(df, test_size=test_size, random_state=42, stratify=df['Label'])
	else:
		train = df.drop_duplicates(subset=['Text'], keep=False)
		test = df[df.duplicated(subset=['Text'], keep=False)]
	train.to_csv(filename.replace('.csv', '_train.csv'), sep=';', index=False)
	test.to_csv(filename.replace('.csv', '_test.csv'), sep=';', index=False)
	logthis.say('Train test set separation done.')

def merge_csv_files(files: List[str], outfile: str) -> None:
	logthis.say('Merging files starts.')
	data: pd.DataFrame = pd.DataFrame([], columns=['Label', 'Repo', 'Text'])
	num_files = len(files)
	for index, file in enumerate(files):
		logthis.say(f'Merging files {100*(index+1)/num_files:.2f}% {index+1}/{num_files}')
		df = pd.read_csv(file, sep=';')
		data = pd.concat([df, data])
	logthis.say(f'Write data to {outfile}')
	data.to_csv(outfile, sep=';', index=False)
	logthis.say('Merging files done.')

def getCategories(base_categories: List[str], all_categories: List[str], additional_categories: List[str]) -> List[str]:
	if all_categories:
		return all_categories.copy()
	elif additional_categories:
		return base_categories.copy() + additional_categories.copy()
	else:
		return base_categories.copy()

if __name__ == "__main__":
	parser = argparse.ArgumentParser(prog="python src/final.py", description='Perform all the methods of the program.')
	subparsers = parser.add_subparsers(dest='command', help="Select the command to perform.")
	
	parser_collect_readmes = subparsers.add_parser('collect_readmes', help="Collect readme files, create dataset.",
		description="Collect readmes from collected urls from given file rows.",
		epilog ="""Example: python3 src/collectreadmes.py --input_mode csvfile --input data/awesome_lists_links/awesome_lists.csv 
					--githublinks_file data/awesome_lists_links/repos1.csv --awesome_list_mode 
					--readme_folder data/awesome_lists_links/readme --outfolder data/new_datasets"""
	)
	parser_collect_readmes.add_argument("--input_mode", required=True, choices=collectreadmes.ReadmeCollector.input_modes, help="Set input mode. The input can be given by a csvfile or an url in comand line.")
	parser_collect_readmes.add_argument("--input", required=True, help="Give the input.")
	parser_collect_readmes.add_argument("--category", required='url' in sys.argv, help="Set category of input url. (Required if url input_mode is used)")
	parser_collect_readmes.add_argument('--awesome_list_mode', action=argparse.BooleanOptionalAction, default=False, help='Set mode of links to awesome list.')
	parser_collect_readmes.add_argument("--githublinks_file", help='Give file to save collected githubs if awesome lists are given.')
	parser_collect_readmes.add_argument('--readme_folder', required=True, help='Path to the folder where readme files will be saved per category.')
	parser_collect_readmes.add_argument('--outfolder', required=True, help='Path to the folder, where database per category will be saved.')
	parser_collect_readmes.add_argument('--redownload', help='Redownload the readmes.', action=argparse.BooleanOptionalAction, default=False)
	parser_collect_readmes.add_argument('--input_delimiter', help='Set delimiter of input csv file (default: ";").', default=';')

	parser_preprocess = subparsers.add_parser('preprocess', help="Preprocess given csv data file.")
	parser_preprocess.add_argument('--preprocess_file', required=True, help='Name of .csv the file with the preprocessed data. The file will be saved in the same filename with "_preprocessed" suffix.')

	parser_train_test_split = subparsers.add_parser('train_test_split', help="Makes train test split on given csv file.")
	parser_train_test_split.add_argument('--train_test_file', required=True, help='Name of the file to split.')
	parser_train_test_split.add_argument('--test_size', default=0.2, type=float, help='Size of the test set (default: 0.2).')

	parser_merge_csv = subparsers.add_parser('merge_csv', help='Merge given csv files into one.')
	parser_merge_csv.add_argument('--files', required=True, nargs="+", help='List of csv files to merge with the same header row and ";" delimiter.')
	parser_merge_csv.add_argument('--outfile', required=True, help='Path to outfile csv file with the results.')

	parser_train_models = subparsers.add_parser('train_models', help='Train the models.')
	parser_train_models.add_argument('--train_set', required=True, help='Name of the csv file containing train set.')
	parser_train_models.add_argument('--results_file', required=True, help='Path to the file where results will be saved.')
	parser_train_models.add_argument('--out_folder', required=True, help='Path to the folder where models will be saved.')
	parser_train_models.add_argument('--evaluation_metric', default='test_f1-score_mean', help='Name of the key for evaliuation (default: "f1-score_overall").')
	parser_train_models.add_argument('--gridsearch', default='nogridsearch', choices=['nogridsearch', 'bestmodel', 'bestsampler', 'bestvectorizer', 'all'], help='Set gridsearch mode. (default: nogridsearch)')
	parser_train_models_categories = parser_train_models.add_mutually_exclusive_group(required=False)
	parser_train_models_categories.add_argument('--all_categories', nargs="+", help=f'List of all categories used. Use only if you want not the basic categories. {BASE_CATEGORIES=}')
	parser_train_models_categories.add_argument('--additional_categories', nargs="+", help=f'List of categories adding to basic categories. {BASE_CATEGORIES=}')

	parser_predict = subparsers.add_parser('predict', help='Predict with the given models.')
	parser_predict.add_argument('--inputfolder', required=True, help='Path of folder with the models.')
	parser_predict.add_argument('--test_set', required=True, help='Name of the csv file containing the test set.')
	parser_predict.add_argument('--outfile', required=True, help='Path to outfile csv file with the results.')

	parser_evaluate = subparsers.add_parser('evaluate', help='Evaluate the predictions.')
	parser_evaluate.add_argument('--inputfile', required=True, help='Path of the csv file with the predictions.')
	parser_evaluate.add_argument('--outfile', required=True, help='Path of the json file to write scores.')
	parser_evaluate_categories = parser_evaluate.add_mutually_exclusive_group(required=False)
	parser_evaluate_categories.add_argument('--all_categories', nargs="+", help=f'List of all categories used. Use only if you want not the basic categories. {BASE_CATEGORIES=}')
	parser_evaluate_categories.add_argument('--additional_categories', nargs="+", help=f'List of categories adding to basic categories. {BASE_CATEGORIES=}')

	args = parser.parse_args()

	if args.command == 'collect_readmes':
		collector = collectreadmes.ReadmeCollector()
		if args.input_mode == 'csvfile':
			collector.addCategoriesFromCsvFile(args.input, args.input_delimiter)
		elif args.input_mode == 'url':
			collector.addCategory(args.category, [args.input])

		if args.awesome_list_mode:
			collector.mapAwesomeListsToGithubLinks()
			if args.githublinks_file:
				collector.dumpGithubLinks(args.githublinks_file)

		if args.redownload:
			collector.downloadReadmeFiles(args.readme_folder)

		collector.createDatabase(args.outfolder, args.readme_folder)

	elif args.command == 'preprocess':
		preprocess_file(args.preprocess_file)
	elif args.command == 'train_test_split':
		train_test_split_file(args.train_test_file, args.test_size)
	elif args.command == 'merge_csv':
		merge_csv_files(args.files, args.outfile)
	if args.command == 'train_models':
		categories = getCategories(BASE_CATEGORIES, args.all_categories, args.additional_categories)
		logthis.say(f"{categories=}")
		if args.gridsearch == 'nogridsearch':
			import train
			train.train_models(args.train_set, args.out_folder, args.results_file, categories, args.evaluation_metric)
		elif args.gridsearch == 'bestvectorizer':
			import Experiments.best_vectorizer
			Experiments.best_vectorizer.train_models(args.train_set, args.test_set, args.out_folder, args.results_file, categories, args.evaluation_metric)
		elif args.gridsearch == 'bestsampler':
			import Experiments.best_sampler
			Experiments.best_sampler.train_models()
		elif args.gridsearch == 'bestmodel':
			import Experiments.best_model
			Experiments.best_model.train_models()
		elif args.gridsearch == 'all':
			#TODO
			pass
	if args.command == 'predict':
		predictor = Predictor(args.inputfolder, args.test_set, args.outfile)
		predictor.predict()
	if args.command == 'evaluate':
		categories = getCategories(BASE_CATEGORIES, args.all_categories, args.additional_categories)
		logthis.say(f"{categories=}")
		evaluator = Evaluator(args.inputfile, set(categories))
		with open(args.outfile, 'w') as f:
			json.dump(evaluator.evaluate(), f, indent=4)
