# uncertainty-ts-forecasting

- We compare the following methods:
  1. Bayesian RNN (WiML 2017) [https://arxiv.org/abs/1704.02798]
  2. Dropout as a Bayesian Approximation (ICML 2016) [https://arxiv.org/abs/1506.02142]

- Datasets:
  1. Air quality dataset [https://archive.ics.uci.edu/ml/datasets/Air+Quality]

- Dropout Experiments:
  Run the UE_TS_dropout.ipynb for detailed dropout experiments on Air Quality Datasets and visualizations.
  
- Bayes by Backprop:
  bayesian_rnn.py and BBBLayers.py files are for the Bayes by Backprop method. Unfortunately, there are some bugs in the training procedure which needs more attention.
  
- Dependencies
  - Python 3.6+
  - PyTorch==1.1
