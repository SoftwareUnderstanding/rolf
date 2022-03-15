import argparse
from pathlib import Path
import requests
import re
import csv
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

parser = argparse.ArgumentParser("Collect readmes from collected urls from given file rows.")
parser.add_argument("filename", help="Name of the file with the links")
args = parser.parse_args()

REGEX_LINK = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"

input_path = Path(args.filename)
category = input_path.name.replace('.csv', '')

with open(input_path) as f:
	input_links = [line.strip() for line in f.readlines()]

def getReadmeUrlFromGithubUrl(github_url: str) -> str:
	return 'https://raw.githubusercontent.com/' + '/'.join(github_url.split('/')[-2:]) + '/master/README.md'

def getRepoNameFromGitHubUrl(url: str) -> str:
	return '_'.join(url.split('/')[-2:])

def getReadmeFromUrlToFile(url: str, folder: Path, collector: set):
	r = requests.get(getReadmeUrlFromGithubUrl(url))
	if r.status_code == 200:
		with open(folder / (getRepoNameFromGitHubUrl(url) + '.txt'), 'w') as f:
			f.write(r.text)
		collector.add(url)

# Collect github urls from input_links

github_links = set()
for link in input_links:
	r = requests.get(getReadmeUrlFromGithubUrl(link))
	r.raise_for_status()
	for found in re.finditer(REGEX_LINK, r.text):
		if 'github.com' in found[0]:
			github_links.add(found[0])

# Collect valid github links with readme and readmes to files

readmes_path = input_path.parent / 'readme' / category
readmes_path.mkdir(parents=True, exist_ok=True)
valid_github_links = set()
with ThreadPoolExecutor(16) as executor:
	executor.map(lambda x : getReadmeFromUrlToFile(x, readmes_path, valid_github_links), github_links)

# Write list of github links with valid readme

writer = csv.DictWriter(open(f'{input_path.parent.as_posix()}/{category}_repos.csv', 'w'), fieldnames=['Label', 'Repo'], delimiter=';')
writer.writeheader()
for url in valid_github_links:
	writer.writerow({'Label': category, 'Repo': url})

# Write category data to csv

fieldnames = ['Label', 'Repo', 'Text']
data = {key: [] for key in fieldnames}
for url in valid_github_links:
	with open( readmes_path / (getRepoNameFromGitHubUrl(url) + '.txt')) as f:
		data['Text'].append('"' + ' '.join([line.replace('\n', ' ') for line in f]) + '"')
	data['Repo'].append(url)
	data['Label'].append(category)

df = pd.DataFrame(data)
df.drop_duplicates(inplace=True)
df.to_csv(f"data/categories/readme_{category}.csv", sep=';', index=False)
