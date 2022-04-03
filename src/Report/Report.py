from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
import pickle
import numpy as np
import pandas as pd


#def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
#def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]
#def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]
#def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]

tn = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[0, 0]
fp = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[0, 1]
fn = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[1, 0]
tp = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[1, 1]

score_metrics = {'acc': accuracy_score,
			   'prec': precision_score,
			   'recall': recall_score,
			   'f1-score': f1_score,
			   'tp': tp, 'tn': tn,
			   'fp': fp, 'fn': fn}

def report(clf, train_name, test_name, x_train, y_train, X_test, y_test, label, name='classifier', cv=5, dict_scoring=None, fit_params=None, save=False):
	'''
		Function create a metric report automatically with cross_validate function.
		@param clf: (model) classifier
		@param x: (list or matrix or tensor) training x data
		@param y: (list) label data 
		@param name: (string) name of the model (default classifier)
		@param cv: (int) number of fold for cross-validation (default 5)
		@param dict_scoring: (dict) dictionary of metrics and names
		@param fit_aparams: (dict) add parameters for model fitting 
		@param save: (bool) determine if the model need to be saved
		@return: (pandas.dataframe) dataframe containing all the results of the metrics 
		for each fold and the mean and std for each of them
	'''
	if dict_scoring!=None:
		score = dict_scoring.copy() # save the original dictionary
	for i in score.keys():
		if len(set(y_train))>2:
			if i in ["prec", "recall", "f1-score"]:
				score[i] = make_scorer(score[i], average = 'weighted') # make each function scorer
			elif i=="roc_auc":
				score[i] = make_scorer(score[i], average = 'weighted', multi_class="ovo",needs_proba=True) # make each function scorer
			else:
				score[i] = make_scorer(score[i]) # make each function scorer
		elif i in ['prec', 'recall', 'f1-score'] :
			score[i] = make_scorer(score[i], pos_label=label) # make each function scorer
		else:
			score[i] = make_scorer(score[i])

	try:
		scores = cross_validate(clf, x_train, y_train, scoring=score,
			cv=cv, return_train_score=False, n_jobs=-1,  fit_params=fit_params)
	except:
		scores = cross_validate(clf, x_train, y_train, scoring=score,
			cv=cv, return_train_score=False,  fit_params=fit_params)

	# Train test on the overall data
	model = clf
	model.fit(x_train, y_train)
	features = model[:-1].get_feature_names_out()
	print(f'{label}: ', file=open("output.txt", "a"))
	for i in features:
		print(f'{i}', file=open("output.txt", "a"))
	y_pred = model.predict(X_test)#>0.5).astype(int)

	if save:
		filename= name+label+".sav"
		pickle.dump(model, open('results/models/'+filename, 'wb'))


	csvFileName = f"{label.lower().replace(' ', '_')}.csv"
	#with open('results/scoreboards/' + csvFileName, 'r') as csvfile:
	#	rownum = len(csvfile.readlines())
	# initialisation 
	res = {'PipelineID' : label,
		   'Pipeline' : name ,
		   'train_set' : train_name,
		   'validation_set' : test_name}
	for i in scores:  # loop on each metric generate text and values
		if i == "estimator": continue
		for j in enumerate(scores[i]):
			res[i+"_cv"+str(j[0]+1)] = j[1]
		res[i+"_mean"] = np.mean(scores[i])
	
	 # add metrics averall dataset on the dictionary 
	
	#print(scores)
	#print(score)
	del scores['fit_time']
	del scores['score_time']
	print(scores)
	#for i in scores:	# compute metrics 
	#	scores[i] = np.append(scores[i] ,score[i.split("test_")[-1]](model, X_test, y_test))
	#	res[i.split("test_")[-1]+'_overall'] = scores[i][-1]
	
	return pd.DataFrame(data=res.values(), index=res.keys()).T, model