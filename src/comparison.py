import csv
from pathlib import Path
from aimmx import AIMMX
import json

def getRepoNameFromGitHubUrl(url: str) -> str:
	return '_'.join(url.split('/')[-2:])

if __name__ == '__main__':
	repos = {}
	aimmx = AIMMX("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")
	reader = csv.DictReader(open("data/readme_new_preprocessed_test.csv"), delimiter=';')
	path = Path("data/aimmx_output.csv")
	new = not path.exists()
	writer = csv.writer(open("data/aimmx_output.csv", 'a+'), delimiter=';')
	if new:
		writer.writeheader()
	outpath = Path('data/aimmx')
	outpath.mkdir(parents=True, exist_ok=True)
	for row in reader:
		print(row['Repo'])
		if row['Repo'] in repos:
			metadata = {'domain': {'domain_type': repos[row['Repo']]}}
			writer.writerow([row['Label'], row['Repo'], metadata['domain']['domain_type'], metadata['domain'].get('task')])
		else:
			try:
				metadata = aimmx.repo_parse(row['Repo'])
				json.dump(metadata, open(outpath / (getRepoNameFromGitHubUrl(row['Repo'] + '.json')), 'w'), indent=4)
				repos[row['Repo']] = metadata['domain']['domain_type']
				writer.writerow([row['Label'], row['Repo'], metadata['domain']['domain_type'], metadata['domain'].get('task')])
			except:
				pass
			
