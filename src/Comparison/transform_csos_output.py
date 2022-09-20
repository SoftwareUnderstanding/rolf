import csv
from collections import defaultdict

reader = csv.DictReader(open('data/comparison_data/csoc_output_all.csv'), delimiter=';')

data = defaultdict(lambda : {'Labels': set(), 'Csoc_predictions' : set()})

for row in reader:
	data[row['Repo']]['Labels'].add(row['Label'])
	data[row['Repo']]['Csoc_predictions'].update(row['CSOS'].split(','))

writer = csv.DictWriter(open('data/comparison_data/csoc_output_all_transformed.csv', 'w'), delimiter=';', fieldnames=['Labels', 'Repo', 'Csoc_predictions'])
writer.writeheader()

for key, val in data.items():
	writer.writerow({
		'Repo': key,
		'Labels': ','.join(val['Labels']),
		'Csoc_predictions': ','.join(val['Csoc_predictions']),
	})
