from cso_classifier import CSOClassifier
import csv
import json
from pathlib import Path

def getRepoNameFromGitHubUrl(url: str) -> str:
	return '_'.join(url.split('/')[-2:])

if __name__ == '__main__':
	cc = CSOClassifier(modules = "both", enhancement = "first", explanation = True, delete_outliers=True)

	reader = csv.DictReader(open("data/readme_new_preprocessed_test.csv"), delimiter=';')
	data = {}
	ind = 0
	for row in reader:
		data[ind] = {
			"title": "",
			"keywords": "",
			"abstract": row['Text'],
			"Label": row['Label'],
			"Repo": row['Repo'],
		}
		ind += 1

	path = Path("data/csoc_output.csv")
	new = not path.exists()
	writer = csv.DictWriter(open(path, 'a+'), delimiter=';', fieldnames=['Label', 'Repo', 'CSOS'])
	if new:
		writer.writeheader()
	outpath = Path('data/csoc')
	outpath.mkdir(parents=True, exist_ok=True)
	results = cc.batch_run(data, workers=8)
	print(results)
	for key, val in results.items():
		writer.writerow({'Label': data[key]['Label'], 'Repo': data[key]['Repo'], 'CSOS': ','.join(val['enhanced'])})
		json.dump(val, open(outpath / (getRepoNameFromGitHubUrl(data[key]['Repo']) + '.json'), 'w'), indent=4)
			
