import csv
from collections import defaultdict

reader = csv.DictReader(open('data/aimmx_output.csv'), delimiter=';')

data = defaultdict(lambda : {'Labels': set(), 'Aimxx_predictions' : set()})

for row in reader:
	data[row['Repo']]['Labels'].add(row['Label'])
	data[row['Repo']]['Aimxx_predictions'].add(row['Aimmx_domain'])
	if row['Aimmx_task']:
		data[row['Repo']]['Aimxx_predictions'].add(row['Aimmx_task'])

writer = csv.DictWriter(open('data/aimmx_output_transformed.csv', 'w'), delimiter=';', fieldnames=['Labels', 'Repo', 'Aimxx_predictions'])
writer.writeheader()

for key, val in data.items():
	writer.writerow({
		'Repo': key,
		'Labels': ','.join(val['Labels']),
		'Aimxx_predictions': ','.join(val['Aimxx_predictions']),
	})
