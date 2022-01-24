# COVID-19 detection from chest X-ray scans

Coronavirus disease 2019 (COVID-19) is a highly infectious disease caused by [severe acute respiratory syndrome coronavirus 2](https://en.wikipedia.org/wiki/Severe_acute_respiratory_syndrome_coronavirus_2). This is an attempt to create neural network models, which can analyze chest X-ray scans to predict whether patient is healthy, infected with COVID-19 or pneumonia.

*Disclaimer -- those models are experiments only and not production ready.*

# Results

## VGG16 Based Model
![VGG16 Based Model confusion matrix](./docs/vgg16_based.png)

*Confusion matrix for VGG16 Based Model on the test dataset.*

__Precision (%)__
| COVID-19 | Normal | Pneumonia |
|----------|--------|-----------|
| 84       | 93     | 80        |

## Simple Residual Model
![Simple Residual Model confusion matrix](./docs/simple_residual.png)

*Confusion matrix for Simple Residual Model on the test dataset.*

__Precision (%)__
| COVID-19 | Normal | Pneumonia |
|----------|--------|-----------|
| 91       | 85     | 77        |

# Setup

### Requirements
* You need to have Kaggle account in order to download *RSNA Pneumonia Detection Challenge* dataset. Follow [these](https://github.com/Kaggle/kaggle-api#api-credentials) instruction for configuration.
* You should have [virtualenv](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) installed.

### Initialize virtualenv
```
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### Download datasets
The current dataset is composed of the following open source chest X-ray datasets:
* https://github.com/ieee8023/covid-chestxray-dataset
* https://github.com/agchung/Figure1-COVID-chestxray-dataset
* https://www.kaggle.com/c/rsna-pneumonia-detection-challenge

Run the following script to download and preprocess those datasets:
```
python script/setup.py
```

### Train models
Run
```
python train/run.py --model_type vgg16_based
```
or
```
python train/run.py --model_type simple_residual
```
depending on a model you want to train.

# References

1. https://arxiv.org/pdf/2003.09871.pdf
2. https://towardsdatascience.com/detection-of-covid-19-presence-from-chest-x-ray-scans-using-cnn-class-activation-maps-c1ab0d7c294b
3. https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035
4. https://arxiv.org/pdf/1409.1556.pdf
5. https://towardsdatascience.com/residual-blocks-building-blocks-of-resnet-fd90ca15d6ec
6. http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

