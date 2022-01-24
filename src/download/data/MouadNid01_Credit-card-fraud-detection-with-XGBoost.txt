# Credit card fraud detection with XGBoost

## Project's goal
It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase. So the project's goal is to build a classifier that tells if a transaction is a fraud or not.

We'll build two classifiers using the same algorithme **XGBoost** the first will be trained on the 75% of the original dataset and tested on the remaining 25%. And the second one will be trained and tested on the same percentages but this time after oversampling the dataset using **SMOTE**.

## Data review
The dataset used in this project is the kaggle Credit card fraud detection dataset. it's available through this link:
* https://www.kaggle.com/mlg-ulb/creditcardfraud 

It contains two-day transactions made on 09/2013 by European cardholders. The dataset is highly unbalanced, with the minority class which is the fraudulent transactions accounting for only 0.17 %.

## Model evaluation
The models performance on the test set are resumed in the table below:
<div align='center'>
  
| Metric | XGBoost without SMOTE | XGBoost with SMOTE |
| -------- | --------------------- | ---------------- |
| **AUPRC** | 87.73 | 99.96 |
| **Precision** | 89.72 | 99.92 |
| **Recall** | 85.71 | 99.99 |
| **Accuracy** | 99.95 | 99.95 |
  
</div>

## Referrences
* https://arxiv.org/pdf/1106.1813.pdf
* https://medium.com/@douglaspsteen/precision-recall-curves-d32e5b290248
* https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/
