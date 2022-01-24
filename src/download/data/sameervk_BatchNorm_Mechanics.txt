Mechanics of Batch Normalization
================================
To understand how batch normalization is performed, I have carried out step-by-step process, and explained with decently sufficient comments, of performing batch normalization on simple input in tensorflow (2.1).

Paper: Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift

		Sergey Ioffe, Christian Szegedy

		https://arxiv.org/abs/1502.03167

1. Writing custom layers in tf.keras for batch normalization 

2. Using tf.nn.batchnormalization function and comparing the results with custom-defined functions in tf to carry out the same

3. Testing on MNIST dataset

4. BatchNorm_experimental-v1: 

	- examining the functionality of the Keras BatchNormalization Layer
	
	- 3 dense layers each followed by 3 Keras BatchNorm layers, and a final dense layer for multiclass classification
	
	- data processed using scikit-learn StandardScaler

	- using tf.data.Dataset with reshuffle after each iteration set to True for training
	
	- 10 epochs only. train acc: 98.84%, val acc: 97.02%, test acc: 97.11%

5. BatchNormCustom_experimental-v1:
	
	- writing and experimenting with custom code for calculating Batch Normalization as implemented in the paper
	
	- the calculation does not implement the *momemtum* factor such as the one defined in the documentation
	
	- comparing the test results against the method using *momentum*
	
	- running for 2 epochs; accuracy ~ 90%
	
6. BatchNormCustom_final-v1:

	- Cleaned up code of BatchNormCustom_experimental-v1
	
	- same architecture and parameters as in BatchNorm_experimental-v1
	
	- 10 epochs 
	
	- Custom code without *momentum* - train acc: 94.95%, val acc: 94.70%, test acc: 94.59%
	
	- Custom code with *momentum* - train acc: 93.87%, val acc: 94.17%, test acc: 93.71%

	 
