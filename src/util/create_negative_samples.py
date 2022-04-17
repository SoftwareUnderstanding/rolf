
from pathlib import Path
import pandas as pd

readmes_path = Path('data/awesome_lists_links/readme/Other')
fieldnames = ['Label', 'Repo', 'Text']
data = {key: [] for key in fieldnames}
i = 0
for file in readmes_path.iterdir():
	i += 1
	with open(file) as f:
		data['Text'].append('"' + ' '.join([line.replace('\n', ' ') for line in f]) + '"')
	data['Repo'].append(f'repo{i}')
	data['Label'].append('Other')
path = Path("data/negative_samples.csv")
df = pd.DataFrame([], columns=fieldnames)
df = pd.concat([df, pd.DataFrame(data)])
df.drop_duplicates(inplace=True)
df.to_csv(path, sep=';', index=False)