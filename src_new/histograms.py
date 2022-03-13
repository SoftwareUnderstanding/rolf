import pandas as pd
import matplotlib.pyplot as plt



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