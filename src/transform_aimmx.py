import csv
from pathlib import Path
import json

def getRepoNameFromGitHubUrl(url: str) -> str:
	return '_'.join(url.split('/')[-2:])

if __name__ == '__main__':

	repos = {}	
	with open('data/csoc_output_transformed.csv') as f:
		reader = csv.DictReader(f, delimiter=';')
		for row in reader:
			repos[row['Repo']] = {'Labels': row['Labels'], 'Predictions': set()}

	path = Path("data/aimmx")
	for input in path.iterdir():
		with open(input) as f:
			data = json.load(f)
		repo = data['definition']['code'][0]['url']
		repos[repo]['Predictions'].add(data['domain']['domain_type'])

	with open('data/aimmx_output_transformed.csv', 'w') as f:
		writer = csv.DictWriter(f, delimiter=';', fieldnames=['Labels', 'Repo', 'Predictions'])
		writer.writeheader()
		for key, val in repos.items():
			writer.writerow({
				'Labels': val['Labels'],
				'Repo': key,
				'Predictions': ','.join(val['Predictions']),
			})
