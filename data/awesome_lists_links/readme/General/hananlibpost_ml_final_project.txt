# Comparing Decision Trees Ensemble Methods 

This project is part of an assignment. 
We compared three ensembles based decision trees to CatBoost over 150 datasets.

The models were trained using Google Colab.

The algorithms we tested were:

1. **mGBDT** Multi-Layered Gradient Boosting Decision Trees [[pdf]](http://papers.nips.cc/paper/7614-multi-layered-gradient-boosting-decision-trees.pdf "[pdf]") [[code]](https://github.com/kingfengji/mGBDT "[code]")
2. **ThunderGBM** Exploiting GPUs for Efficient Gradient Boosting Decision Tree Training [[pdf]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8727750 "[pdf]") [[code]](https://github.com/Xtra-Computing/thundergbm "[code]")
3. **NODE**: Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data [[pdf]](https://arxiv.org/pdf/1909.06312.pdf "pdf") [[code]](https://github.com/Qwicen/node "code")
4.  **CatBoost** CatBoost: unbiased boosting with categorical features [[pdf]](https://arxiv.org/pdf/1706.09516.pdf "pdf") [[code]](https://github.com/catboost/catboost "code")


## Running the code

The notebooks folder contains notebooks to run the code. All the code was trained using Google Colab.
1. Open google colab
2. Upload the notebook
3. Add the path to the dataset in the places where is it mentioned.
4. Add a path to save the results



### Hyperparameters
Hyperparameters values used to tune the model

mGBDT : {'num_boosted_round': [1, 8], 'max_depth' : [2, 7]} <br/>
ThunderGBM : {'depth' : [4,10], 'n_trees' :[10, 80]} <br/>
NODE: {'depth' : [2, 8], 'num_layers' : [2,4]} <br/>
CatBoost {'depth': [2, 4], 'learning_rate' : [0.1, 0.3], 'l2_leaf_reg' : [10,20]} <br/>

### Metrics
The algorithms were tested on the following metrics:
- TPR - True positive rate
- FPR - False positive rate
- Precision 
- ROC-AUC - The area under the ROC curve
- Precision-Recall AUC - the area under the precision-recall curve
- Training time in seconds
- Inference time of 1000 records measured in seconds

For multiclass datasets, we calculate the average score of each metric without weights.


## **Papers**:
Feng, J., Yu, Y. and Zhou, Z.H., 2018. Multi-layered gradient boosting decision trees. In Advances in neural information processing systems (pp. 3551-3561). [pdf](http://papers.nips.cc/paper/7614-multi-layered-gradient-boosting-decision-trees.pdf "[pdf]")

Wen, Z., Shi, J., He, B., Chen, J., Ramamohanarao, K. and Li, Q., 2019. Exploiting GPUs for efficient gradient boosting decision tree training. IEEE Transactions on Parallel and Distributed Systems, 30(12), pp.2706-2717. [pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8727750 "[pdf]")

Popov, S., Morozov, S. and Babenko, A., 2019. Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data. arXiv preprint arXiv:1909.06312. [pdf](https://arxiv.org/pdf/1909.06312.pdf "pdf")

Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A.V. and Gulin, A., 2018. CatBoost: unbiased boosting with categorical features. In Advances in neural information processing systems (pp. 6638-6648). [pdf](https://arxiv.org/pdf/1706.09516.pdf "pdf")



