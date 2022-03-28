import pandas as pd
import matplotlib.pyplot as plt


import matplotlib.pyplot as plt

df = pd.read_csv('data/readme_new_preprocessed_train.csv', sep=';')
#df.drop_duplicates('Text', inplace=True, keep=False)
print(df.shape)

#df = pd.read_csv('data/readme.csv', sep=';')
df.drop('Text', axis=1, inplace=True)
distribution = df.groupby('Label').count()
distribution.sort_values('Repo', ascending=[False], inplace=True)
print(distribution)

labels = [key for key, _ in distribution.iterrows()]
print(labels)
vals = distribution['Repo'].values
print(vals)

plt.figure(figsize=(30,15))
plt.bar(labels, vals, color='#00c7c3')
plt.title(f'Distribution of samples between categories', size=18, weight='bold')
plt.xlabel('Categories', size=16, weight='bold')
plt.ylabel('Frequency', size=16, weight='bold')
plt.xticks(rotation=0, size=16)
plt.yticks(size=14)
plt.legend(prop={'size': 14})
plt.savefig('results/pics/class_distribution_unique_train.png')
plt.close()


exit(0)
#plt.xticks(fontsize=4)

categories = ["Audio", "Computer Vision","Graphs", "Natural Language Processing", "Reinforcement Learning", "Sequential"]
colors = ['red', 'blue', 'pink', 'purple', 'yellow', 'green']
def create_hist(df):
	plt.figure(figsize=(20, 10))
	df = df[['PipelineID', 'f1-score_overall']]
	for i in range(len(categories)):
		df1 = df[df['PipelineID'].str.contains(categories[i])]['f1-score_overall'].astype(str).str[:4].astype(float)
		print(df1[:4])
		plt.hist(df1, bins=10, color=colors[i], alpha=0.7, label=categories[i], width=0.01)
	plt.xlim(xmin=0.0, xmax = 1.0)
	plt.title('Final results structured random undersampling, preprocessed')
	plt.legend()
	

df = pd.read_csv('final_results_structured_sampling_preprocessed.csv', delimiter=';')
create_hist(df)
plt.savefig('results/pics/results_structured_sampling_preprocessed.csv.png')
plt.show()