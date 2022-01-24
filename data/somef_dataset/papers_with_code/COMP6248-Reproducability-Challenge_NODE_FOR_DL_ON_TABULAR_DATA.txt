# NODE_FOR_DL_ON_TABULAR_DATA
 For COMP6248-Reproducability-Challenge

Team member:Xinxing Cheng, Jiajie Chen, Shengyi Yang

The reproducted paper from: https://arxiv.org/abs/1909.06312

Improvements: 
Rewirte the data loader
Custom dataset class
Change code by using torchbearer to evalute model(call function to save best model, and loss history)
Add tensorboard for dynamic visualisation
function to save data set to pickle

Experiments:
NODE(both shallow and deep), catboost, Xgboost
for data set:['A9A','PROTEIN','YEAR','MICROSOFT','YAHOO','CLICK']
'EPSILON' and 'HIGGS' are highly GPU momery usage.

we added two more date set 
1. ‘ADULT’: https://www.kaggle.com/uciml/adult-census-income
2. ‘SHELTER':https://www.kaggle.com/c/shelter-animal-outcomes/data
