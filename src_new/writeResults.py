import csv
import pandas as pd

def writeResults(results_filename : str, df_results : pd.DataFrame, train : str, test : str):
	with open('results/' + results_filename, 'a+') as csvfile:
		csvWriter = csv.writer(csvfile, delimiter=';')
		csvWriter.writerow([df_results['PipelineID'].iloc[-1],
							df_results['Pipeline'].iloc[-1],
							df_results['test_acc_mean'].iloc[-1],
							df_results['test_prec_mean'].iloc[-1],
							df_results['test_recall_mean'].iloc[-1],
							df_results['test_f1-score_mean'].iloc[-1],
							df_results['acc_overall'].iloc[-1],
							df_results['prec_overall'].iloc[-1],
							df_results['recall_overall'].iloc[-1],
							df_results['f1-score_overall'].iloc[-1],
							train,
							test])
