import argparse
from collections import defaultdict
from pathlib import Path
import sys
from typing import Set
import logthis
import requests
import re
import csv
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

#python3 src/collect_readmes.py --input_mode csvfile --input data/awesome_lists_links/awesome_lists.csv --githublinks_file data/awesome_lists_links/repos1.csv --link_mode awesomelist --readme_folder data/awesome_lists_links/readme/ --outfile data

REGEX_LINK = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"

class ReadmeCollector:

	input_modes = ['csvfile', 'url']
	link_modes = ['github', 'awesomelist']

	def __init__(self):
		self.__categories = defaultdict(list)
		
	def __getReadmeUrlFromGithubUrl(self, github_url: str) -> str:
		return 'https://raw.githubusercontent.com/' + '/'.join(github_url.split('/')[-2:]) + '/master/README.md'

	def __getReadme(self, url: str) -> requests.Response:
		readme_url = self.__getReadmeUrlFromGithubUrl(url)
		r = requests.get(readme_url)
		if r.status_code != 200:
			r = requests.get(readme_url.replace('README', 'readme'))
		return r

	def __getReadmeFromUrlToFile(self, url: str, folder: Path) -> None:
		r = self.__getReadme(url)
		if r.status_code == 200:
			with open(folder / (self.getRepoNameFromGitHubUrl(url) + '.txt'), 'w') as f:
				f.write(r.text)
		logthis.say(f"Downloading: {folder} {url}")

	def collectReadmesFromFolder(self, folder_path: str, category: str) -> None:
		path = Path(folder_path)
		for file in path.iterdir():
			with open(file.absolute()) as f:
				text = ' '.join(f.readlines())
				if text == "404: Not Found":
					continue
				text = text.replace('\n', ' ')
				self.__categories[category].append(text)

	def getRepoNameFromGitHubUrl(self, url: str) -> str:
		return '_'.join(url.split('/')[-2:])

	# Add new category with links to category list
	def addCategory(self, category: str, links: list[str]) -> None:
		self.__categories[category.replace('/', '_')] = links

	# Get categories and links from given csv file
	def addCategoriesFromCsvFile(self, path: str, delimiter: str = ';') -> None:
		with open(path) as f:
			reader = csv.DictReader(f, delimiter=delimiter)
			for row in reader:
				self.__categories[row['Label'].replace('/', '_')].append(row['Repo'])
		
	# Collects github links per categories from awesome links
	def collectCategoriesFromAwesomeLists(self) -> None:
		for category in self.__categories.keys():
			with ThreadPoolExecutor() as executor:
				res = executor.map(self.collectGithubUrlsFromAwesomeList, self.__categories[category])
				links = set()
				for r in res:
					links = links.union(r)
			self.__categories[category] = list(links)

	# Collect github urls from input_link
	def collectGithubUrlsFromAwesomeList(self, input_url: str) -> Set[str]:
		github_urls = set()
		r = self.__getReadme(input_url)
		try:
			r.raise_for_status()
			for found in re.finditer(REGEX_LINK, r.text):
				if 'github.com' in found[0]:
					github_urls.add(found[0])
		except:
			pass
		return github_urls
	
	# Collect valid github links with readme and write readmes to files
	def downloadReadmeFiles(self, readmes_folder: str) -> None:
		for category, github_links in self.__categories.items():
			readmes_path = Path(readmes_folder) / category
			readmes_path.mkdir(parents=True, exist_ok=True)
			with ThreadPoolExecutor() as executor:
				executor.map(lambda x: self.__getReadmeFromUrlToFile(x, readmes_path), github_links)

	# Write list of github links
	def dumpGithubLinks(self, outfile: str):
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

	# Create database and save to csv
	def createDatabase(self, outfolder: str, readmes_folder: str) -> None:
		readmes_path = Path(readmes_folder)
		readmes_path.mkdir(parents=True, exist_ok=True)
		fieldnames = ['Label', 'Repo', 'Text']
		for category, links in self.__categories.items():
			data = {key: [] for key in fieldnames}
			for url in links:
				readme_path = Path(readmes_path / category / (self.getRepoNameFromGitHubUrl(url) + '.txt'))
				if readme_path.exists():
					with open(readme_path) as f:
						data['Text'].append('"' + ' '.join([line.replace('\n', ' ') for line in f]) + '"')
					data['Repo'].append(url)
					data['Label'].append(category)
			path = Path(f"{outfolder}/readme_{'_'.join(category.lower().split())}.csv")
			path.parent.mkdir(parents=True, exist_ok=True)
			if path.exists():
				df = pd.read_csv(path, sep=';')
			else:
				df = pd.DataFrame([], columns=fieldnames)
			df = pd.concat([df, pd.DataFrame(data)])
			df.drop_duplicates(inplace=True)
			df.to_csv(path, sep=';', index=False)

	# Clear data
	def clear(self):
		self.__categories.clear()

if __name__ == "__main__":

	parser = argparse.ArgumentParser("python src/collect_readmes.py", description="Collect readmes from collected urls from given file rows.")
	
	parser.add_argument("--input_mode", help="Set input mode.", required=True, choices=ReadmeCollector.input_modes)
	parser.add_argument("--category", help="Set category of input url. (Required if url is given)", required='url' in sys.argv)

	parser.add_argument("--input", help="Set input.", required=True)
	
	parser.add_argument('--link_mode', help='Set mode of links.', required=True, choices=ReadmeCollector.link_modes)
	parser.add_argument("--githublinks_file", help='Give file to save collected githubs from awesome lists.', required='awesomelist' in sys.argv)
	
	parser.add_argument('--readme_folder', required=True, help='Path to the folder where readme files will be saved per category.')
	parser.add_argument('--outfolder', required=True, help='Path to the folder, where database per category will be saved.')
	parser.add_argument('--redownload', help='Do not redownload the readmes.', action=argparse.BooleanOptionalAction, default=False)
	parser.add_argument('--input_delimiter', help='Set delimiter of input csv file (optional, default=";").', default=';')
	
	args = parser.parse_args()

	collector = ReadmeCollector()

	if args.input_mode == 'csvfile':
		collector.addCategoriesFromCsvFile(args.input, args.input_delimiter)
	elif args.input_mode == 'url':
		collector.addCategory(args.category, [args.input])

	if args.link_mode == 'awesomelist':
		collector.collectCategoriesFromAwesomeLists()
		collector.dumpGithubLinks(args.githublinks_file)

	if args.redownload:
		collector.downloadReadmeFiles(args.readme_folder)

	collector.createDatabase(args.outfolder, args.readme_folder)
