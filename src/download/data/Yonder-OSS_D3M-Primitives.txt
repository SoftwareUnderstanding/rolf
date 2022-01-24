# Repository of Yonder Open Source D3M Primitives

## Data Preprocessing

1. **DataCleaningPrimitive**: wrapper for the data cleaning primitive based on the Yonder [punk](https://github.com/NewKnowledge/punk) library.

2. **DukePrimitive**: wrapper of the [Duke library](https://github.com/NewKnowledge/duke) in the D3M infrastructure.

3. **SimonPrimitive**: LSTM-FCN neural network trained on 18 different semantic types, which infers the semantic type of each column. Base library: https://github.com/NewKnowledge/simon/tree/d3m-simon

4. **GoatForwardPrimitive**: geocodes names of locations into lat/long pairs with requests to photon geocoding server (based on OpenStreetMap)

5. **GoatReversePrimitive**: geocodes lat/long pairs into geographic names of varying granularity with requests to photon geocoding server (based on OpenStreetMap)

## Clustering

1. **HdbscanPrimitive**: wrapper for *scikit-learn*'s HDBSCAN and DBSCAN implementations

2. **StorcPrimitive**: wrapper for *tslearn*'s kmeans implementations

3. **SpectralClustering**: wrapper for *scikit-learn*'s Spectral Clustering implementation

## Feature Selection

1. **PcaFeaturesPrimitive**: wrapper of the [Punk](https://github.com/NewKnowledge/punk) feature ranker into D3M infrastructure.

2. **RfFeaturesPrimitive** wrapper of the [Punk](https://github.com/NewKnowledge/punk) punk rrfeatures library into D3M infrastructure

## Dimensionality Reduction

1. **TsnePrimitive**: wrapper for *scikit-learn*'s TSNE implementation

## Natural Language Processing

1. **Sent2VecPrimitive**: converts sentences into numerical feature representations. Base library: https://github.com/NewKnowledge/nk-sent2vec

## Image Classification

1. **GatorPrimitive**: Inception V3 model pretrained on ImageNet finetuned for classification

**imagenet.py**: ImagenetModel class with finetune() and finetune_classify() methods

## Object Detection

1. **ObjectDetectionRNPrimitive**: wrapper for the Keras implementation of Retinanet from [this repo](https://github.com/fizyr/keras-retinanet). The original Retinanet paper can be found [here](https://arxiv.org/abs/1708.02002).

## Time Series Classification

1. **KaninePrimitive**: wrapper for tslearn's KNeighborsTimeSeriesClassifier algorithm 

2. **LstmFcnPrimitive**: wrapper for LSTM Fully Convolutional Networks for Time Series Classification paper, original repo (https://github.com/titu1994/MLSTM-FCN), paper (https://arxiv.org/abs/1801.04503)

**layer_utils.py**: implementation of AttentionLSTM in tensorflow (compatible with 2), originally from https://github.com/houshd/LSTM-FCN

**lstm_model_utils.py**: functions to generate LSTM_FCN model architecture and data generators

**var_model_utils.py**: wrapper of the *auto_arima* method from *pmdarima.arima* with some specific parameters fixed

## Time Series Forecasting

1. **DeepArPrimitive**: DeepAR recurrent, autoregressive Time Series Forecasting algorithm (https://arxiv.org/abs/1704.04110). Base library: https://github.com/NewKnowledge/deepar

2. **VarPrimitive**: wrapper for *statsmodels*' implementation of vector autoregression for multivariate time series

**var_model_utils.py**: wrapper of the *auto_arima* method from *pmdarima.arima* with some specific parameters fixed

## Interpretability

**shap_explainers**: wrapper of Lundberg's shapley values implementation for tree models. Currently integrated into *d3m.primitives.learner.random_forest.DistilEnsembleForest* as *produce_shap_values()*

