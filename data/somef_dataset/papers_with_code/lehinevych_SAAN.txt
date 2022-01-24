
# SAAN
This is an implementation of the Self-attention aggregation network as described in the [SAAN](https://arxiv.org/abs/2010.05340) paper. 

SAAN is a neural network architecture that solves face template aggregation problems using self-attention mechanisms. 
Particularly, we employ [Transformer](https://github.com/tensorflow/tensor2tensor) implementation for the sequence encoding.

## Contents
  * [Contents](#contents)
   * [Detailed information](#detailed-information)
   * [Results on benchmarks](#model-training-and-evaluation)

## Detailed information


1. ### Model training and evaluation

   There are two notebooks which demonstrate [single](Single_identity_aggregation_and_attention_dist.ipynb) and 
   [multi-identity](Multi_identity_aggregation_model_for_repo.ipynb) aggregation models with the respective training and validation pipelines created using tf.estimator and tf.Dataset APIs. The aggregation architecture is shared and could be found in [aggregator.py](aggregator_utils/aggregator.py). 


   [config.py](aggregator_utils/config.py) specifies  the configurations of the data sampler and different aggregators.


2. ### Results on benchmarks
   [IJB notebook](IJB_preprocessing_and_becnhmarking_for_repo.ipynb) contains the complete report with the respective visualization on identification and verificaiton metrics using IJB-C benchmark.

**Note**: It is advised to download the repository and display the .html files via the browsers in order to view the results.  
   
