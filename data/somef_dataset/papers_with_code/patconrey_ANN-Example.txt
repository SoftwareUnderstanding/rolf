# ANN-Example
This is an example script to create, train, and evaluate an artificial neural network.

# Problem Statement
A bank is losing customers at an alarming rate. They want to investigate why this is happening and identify which
customers are at a higher risk of leaving. If they can identify those customers, then they can further analyze
trends within that customer segment. At the same time, they'll be able to take precautions by reaching out to those
high-risk customers. So, our job is to predict whether or not a customer will leave the bank, giving a probability between 0 and 1.

# Performance
The trained ANN scores a mean accuracy of 85.5% over 10 folds, each with 100 epochs. The variance of the accuracies 
collected by the 10-fold cross validation was 1.12%.

# Data Set
In an effort to determine the high-risk customers, the bank observed 10,000 customers over six months. The dataset was partitioned into a training set of 8,000 samples and a testing set of 2,000 samples. They collected
a variety of datapoints they figured would be indicative of retention (or, more technically, "churn"). An excerpt of 
the dataset is provided below

|RowNumber   	| CustomerID  	| Surname  	| CreditScore  	| Geography  	| Gender  	| Age  	| Tenure  	| Balance  	| NumOfProducts  	| HasCrCard  	| IsActiveMember  	| EstimatedSalary  	| Exited  	|
|---	        |---	          |---	      |---	          |---	        |---	      |---	  |---	      |---	      |---	            |---	        |---	              |---	              |---	      |
| 1   	      | 15634602  	  | Hargrave  | 619   	      | France    	| Female  	| 42   	| 2       	| 0       	| 1             	| 1          	| 1               	| 101348.88       	| 1  	      |
| 49   	      | 15766205  	  | Yin       | 550   	      | Germany    	| Male    	| 38   	| 2       	| 103391.38 | 1             	| 0          	| 1               	| 90878.13         	| 0  	      |

I didn't collect this dataset. So far as I know, it was generated. If you're interested in obtaining the full dataset, please reach out to me and I can send it to you. I won't be hosting it in this repository to protect the work of its creator.

# General Architecture
- Input layer with 11 features
- First Hidden Layer with 6 nodes & PReLU activation
- Dropout with 10%
- Second Hidden Layer with 6 nodes & PReLU activation
- Dropout with 10%
- Output layer with sigmoid activation

- OPTIMIZER: adam
- LOSS FN: binary cross entropy
- BATCH SIZE: 10
- EPOCHS: 100

# Environment
Python 3.6.6

Keras 2.2.4

Tensorflow 1.10.0

# Improvements

So far, we've statistically improved accuracy by 2% by implementing the layers with PReLU activations. For more information, read this _amazing_ paper:

https://arxiv.org/abs/1502.01852

# Future Work
We want to create an ROC curve for each parameter tuning and use its AUC measure to evaluate the relative performance 
between hyperparameter selections. It would also be interesting to evaluate different architectures, adding another 
hidden layer and increasing the dropout rate of each layer.
