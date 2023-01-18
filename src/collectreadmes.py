import argparse
from collections import defaultdict
from pathlib import Path
import sys
from typing import Iterable, List, Set, Union
import logthis
import requests
import re
import csv
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

REGEX_LINK = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
LOG = logthis.say

class ReadmeCollector:

	"""
	This class can be used to create a database for training by collecting readme texts of github repositories.
	The class can also handle awesome list links and collect all the github repositories found in their readme text.
	The output contains all the readme texts in separate files for reach repository and the created database.
	If awesome lists are provided, the collected github urls are saved as well.
	"""
	input_modes = ['csvfile', 'url']

	def __init__(self):
		self.__categories: defaultdict[str, List[str]] = defaultdict(list)

	def __getReadmeUrlFromGithubUrl(self, github_url: str) -> str:
		"""
		Generates the url of readme file for the given github url.

		Params
		----------
		github_url: (str) The github url.

		Return
		----------
		(str) The url of the readme file.
		"""
		return 'https://raw.githubusercontent.com/' + '/'.join(github_url.split('/')[-2:]) + '/master/README.md'

	def __getReadmeText(self, github_url: str) -> Union[str, None]:
		"""
		Gets the text for the given github repo or None if not found by the method.
		Tries 'README.md' and 'readme.md' files in root folder of repo.

		Params
		-----------
		github_url: (str) The github url.

		Return
		----------
		(str) The text of the readme or None if not found.
		"""
		readme_url = self.__getReadmeUrlFromGithubUrl(github_url)
		r = requests.get(readme_url)
		res = None
		if r.status_code == 200:
			res = r.text
		else:
			r = requests.get(readme_url.replace('README', 'readme'))
			if r.status_code == 200:
				res = r.text
		return res

	def __downloadReadmeTextToFile(self, github_url: str, folder: Path) -> None:
		"""
		Downloads the readme file of the given repo and writes in the given folder into '{repo_name}.txt' file.

		Params
		---------
		github_url: (str) The github url.
		folder: (str) Path to the folder where readme text will be saved.
		"""
		text = self.__getReadmeText(github_url)
		if text is not None:
			with open(folder / (self.getRepoNameFromGitHubUrl(github_url) + '.txt'), 'w') as f:
				f.write(text)
		LOG(f"Downloaded: {github_url=} to {folder=}")

	def getRepoNameFromGitHubUrl(self, url: str) -> str:
		"""
		Generates a repository name from the given github url.

		Params
		--------
		url: (str) The url to generate the name from.

		Return
		--------
		(str) The generated repository name.
		"""
		return '_'.join(url.split('/')[-2:])

	def addCategory(self, category: str, links: Iterable[str]) -> None:
		"""
		Function to add a sinle category's links to the cateories Dict.

		Params
		--------
		category: (str) Name of the category
		links: (iterable[str]) Links to add to category
		"""
		LOG(f"Adding {category=} with given github_urls, num: {len(links)}")
		self.__categories[category.replace('/', '_')].extend(links)

	def addCategoriesFromCsvFile(self, path: str, delimiter: str = ';') -> None:
		"""
		Function to add every category and it's link to the categories Dict.
		At this step the links could be awesome list links or github links.

		Params
		---------
		path: (str) Path to the csv file containing the data with 'Label', 'Repo' headers
			Examples: 'data/awesome_lists_links/awesome_lists.csv', 'data/awesome_lists_links/repos.csv'.
		delimiter: (str) Delimiter of the csv file.
		"""
		LOG("Adding categories from csvfile.")
		with open(path) as f:
			reader = csv.DictReader(f, delimiter=delimiter)
			for row in reader:
				self.__categories[row['Label'].replace('/', '_')].append(row['Repo'])
		LOG(f"Added categories: {list(self.__categories.keys())}")

	def mapAwesomeListsToGithubLinks(self) -> None:
		"""
		If awesome list(s) were given in the input, then we have to map those to github links (does in-place).
		"""
		LOG("Mapping awesome lists to github urls.")
		for category in self.__categories.keys():
			LOG(f"Mapping {category=}")
			with ThreadPoolExecutor() as executor:
				res = executor.map(self.__collectGithubUrlsFromAwesomeList, self.__categories[category])
				links = set()
				for r in res:
					links = links.union(r)
			self.__categories[category] = list(links)

	def __collectGithubUrlsFromAwesomeList(self, input_url: str) -> Set[str]:
		"""
		Collects github urls from the given awesome list link (or any link).

		Params
		----------
		input_url: (str) The url where the github urls will be collected from.

		Return
		----------
		(Set[str]) Set of the github urls collected.
		"""
		github_urls = set()
		text = self.__getReadmeText(input_url)
		if text is not None:
			for found in re.finditer(REGEX_LINK, text):
				if 'github.com' in found[0]:
					github_urls.add(found[0])
		return github_urls

	def downloadReadmeFiles(self, readmes_folder: str) -> None:
		"""
		Downloads all the readme files per category. Creates a folder for each category in the given folder.

		Params
		---------
		readmes_folder: (str) Path to the folder where the readmes will be saved.
			Example: 'data/awesome_lists_links/readme'
		"""
		LOG(f"Downloading readme files to {readmes_folder=}")
		for category, github_links in self.__categories.items():
			readmes_path = Path(readmes_folder) / category
			readmes_path.mkdir(parents=True, exist_ok=True)
			LOG(f"Downloading readme files for {category=} to '{readmes_path.as_posix()}'")
			with ThreadPoolExecutor() as executor:
				executor.map(lambda x: self.__downloadReadmeTextToFile(x, readmes_path), github_links)

	def dumpGithubLinks(self, outfile: str) -> None:
		"""
		Dumps (in append mode) github links into a csvfile. Fields: ('Label': category, 'Repo': github url).

		Params
		---------
		outfile: (str) Path to the file where links will be dumped.
			Example: 'data/awesome_lists_links/respo.csv'.
		"""
		LOG(f"Writing github links to '{outfile}'")
		path = Path(outfile)
		path.parent.mkdir(parents=True, exist_ok=True)
		fieldnames = ['Label', 'Repo']
		data = {key: [] for key in fieldnames}
		for category, links in self.__categories.items():
			for link in links:
				data['Label'].append(category)
				data['Repo'].append(link)
		if path.exists():
			df = pd.read_csv(path, sep=';')
		else:
			df = pd.DataFrame([], columns=fieldnames)
		df = pd.concat([df, pd.DataFrame(data)])
		df.drop_duplicates(inplace=True)
		df.to_csv(path, sep=';', index=False)

	def createDatabase(self, outfolder: str, readmes_folder: str) -> None:
		"""
		Looks for the readme files containing readme text linked to the github urls stored in __categories.
		Generates a database per category from the readme texts ready for preprocessing or training.
		If database exists, only appends the new readme texts.

		Params
		---------
		outfolder: (str) The folder where the databases are stored per category folders.
			Example: 'data/new datasets'
		readmes_folder: (str) Path to the folder where readmes are stored in separate folders per category.
			Example: 'data/awesome_lists_links/readme'
		"""
		LOG(f"Generating database to '{outfolder}'")
		readmes_folder_path = Path(readmes_folder)
		fieldnames = ['Label', 'Repo', 'Text']
		for category, links in self.__categories.items():
			path = Path(f"{outfolder}/{category}/readme_{'_'.join(category.lower().split())}.csv")
			path.parent.mkdir(parents=True, exist_ok=True)
			LOG(f"Generating database for {category=} to '{path.as_posix()}'")
			data = {key: [] for key in fieldnames}
			for url in links:
				readme_path = Path(readmes_folder_path / category / (self.getRepoNameFromGitHubUrl(url) + '.txt'))
				if readme_path.exists():
					with open(readme_path) as f:
						data['Text'].append('"' + ' '.join([line.replace('\n', ' ') for line in f]) + '"')
					data['Repo'].append(url)
					data['Label'].append(category)
			if path.exists():
				df = pd.read_csv(path, sep=';')
			else:
				df = pd.DataFrame([], columns=fieldnames)
			df = pd.concat([df, pd.DataFrame(data)])
			df.drop_duplicates(inplace=True)
			df.to_csv(path, sep=';', index=False)

	def clear(self) -> None:
		"""
			Resets the class by clearing it's data.
		"""
		self.__categories.clear()

if __name__ == "__main__":

	parser = argparse.ArgumentParser("python src/collect_readmes.py",
		description="Collect readmes from collected urls from given file rows.",
		epilog ="""Example: python3 src/collectreadmes.py --input_mode csvfile --input data/awesome_lists_links/awesome_lists.csv
					--githublinks_file data/awesome_lists_links/repos1.csv --awesome_list_mode
					--readme_folder data/awesome_lists_links/readme --outfolder data/new_datasets""")

	parser.add_argument("--input_mode", required=True, choices=ReadmeCollector.input_modes, help="Set input mode. The input can be given by a csvfile or an url in comand line.")
	parser.add_argument("--input", required=True, help="Give the input.")
	parser.add_argument("--category", required='url' in sys.argv, help="Set category of input url. (Required if url input_mode is used)")

	parser.add_argument('--awesome_list_mode', action=argparse.BooleanOptionalAction, default=False, help='Set mode of links to awesome list.')
	parser.add_argument("--githublinks_file", help='Give file to save collected githubs if awesome lists are given.')

	parser.add_argument('--readme_folder', required=True, help='Path to the folder where readme files will be saved per category.')
	parser.add_argument('--outfolder', required=True, help='Path to the folder, where database per category will be saved.')
	parser.add_argument('--redownload', help='Redownload the readmes.', action=argparse.BooleanOptionalAction, default=False)
	parser.add_argument('--input_delimiter', help='Set delimiter of input csv file (default: ";").', default=';')

	args = parser.parse_args()

	collector = ReadmeCollector()

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
