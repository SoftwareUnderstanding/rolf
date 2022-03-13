import csv
import sys
import json
from pathlib import Path

path = Path(sys.argv[1])
reader = csv.DictReader(open(path), delimiter=';')
res = {}
for row in reader:
	if (row['train_set'], row['validation_set']) not in res:
		res[(row['train_set'], row['validation_set'])] = {}
	if row['PipelineID'] not in res[(row['train_set'], row['validation_set'])]:
		res[(row['train_set'], row['validation_set'])][row['PipelineID']] = row
	elif res[(row['train_set'], row['validation_set'])][row['PipelineID']]['test_f1-score_mean'] < row['test_f1-score_mean']:
		res[(row['train_set'], row['validation_set'])][row['PipelineID']] = row
		res[(row['train_set'], row['validation_set'])][row['PipelineID']]['preprocessing'] = 'Preprocessed'

for (train_set, validation_set), val in res.items():
	writer = csv.DictWriter(open(f'results/results/{train_set[:-4]}_{validation_set}', 'a+'), delimiter=';', fieldnames=reader.fieldnames+['preprocessing'])
	writer.writeheader()
	vals = (v for v in val.values())
	writer.writerows(vals)

#for key, val in res.items():
#	print(key)
#	print(json.dumps(val, indent=4))

