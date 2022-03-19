# dsilt-tsa
Code and materials for the Data Science in Layman's Terms: Time Series Analysis course

Before running any of the code, be sure to install the requirements.  Links to the datasets are below.

```
python -m virtualenv env
source env/bin/activate
python -m pip install -r requirements.txt
```

### Time Series Classification API Service

For the time_series_classification API service, you will need to train the models before starting the API.  This can be done by running the evaluation script:

```
python evaluate_models.py
```

The API can then be ininialized with:

```
uvicorn main:app
```

Go to http://127.0.0.1:8000/docs to try it out.

# Links

Links from the lecture slides are listed below, under the name of the lecture they are from.

### Autocorrelation

* Create the sine/cosine graphs from the cycles/seasonality slide:  https://www.desmos.com/calculator/nqfu5lxaij

### Forecasting with Nonlinear Models

* Convolutional neural networks learn different features at different levels:  https://indico.io/blog/exploring-computer-vision-convolutional-neural-nets/
* Paper that introduced the transformer: https://arxiv.org/abs/1706.03762

### Signals

* Visual explanation of the Fourier Transform:  https://www.youtube.com/watch?v=spUNpyF58BY

### Anomaly Detection and Forecasting Project

* Kaggle notebook:  https://www.kaggle.com/nicholaslincoln/anomaly-detection-forecasting

### Segmentation Project

* Kaggle notebook:  https://www.kaggle.com/nicholaslincoln/brett-favre-change-point-detection

### Sequence Prediction Project

* Maestro MIDI Dataset:  https://magenta.tensorflow.org/datasets/maestro
* Transformer Music Generation repo:  https://github.com/nlinc1905/transformer-music-generation

### Time Series Classification Project

* ECG dataset from Kaggle:  https://www.kaggle.com/shayanfazeli/heartbeat
* Paper that describes the dataset and pre-processing steps:  https://arxiv.org/pdf/1805.00794.pdf

### Signal Processing Project

* LIGO home:  https://www.ligo.caltech.edu/
* Dataset source:  https://www.gw-openscience.org/catalog/GWTC-1-confident/single/GW150914/
* YouTube video that plays the output wav file:  https://www.youtube.com/watch?v=IYq39kCjUns

### Time and Location Project

* San Francisco crime data 2003-2018:  https://data.sfgov.org/Public-Safety/Police-Department-Incident-Reports-Historical-2003/tmnf-yvry
* San Francisco crime data 2018-present:  https://data.sfgov.org/Public-Safety/Police-Department-Incident-Reports-2018-to-Present/wg3w-h783
* GeoJSON for police districts:  https://data.sfgov.org/Public-Safety/Current-Police-Districts/wkhw-cjsf

### What Else About Time Series

* Data Science in Layman's Terms: Statistics:  https://www.amazon.com/Data-Science-Laymans-Terms-Statistics/dp/0692150757
* Quantstart Advanced Algorithmic Trading ebook:  https://www.quantstart.com/advanced-algorithmic-trading-ebook/
* Jason Brownlee's time series forecasting book:  https://machinelearningmastery.com/introduction-to-time-series-forecasting-with-python/
* Jason Brownlee's guide to data transformation for time series modeling: https://machinelearningmastery.com/time-series-forecasting-supervised-learning/
* Jason Brownlee's guide to data transformation for time series modeling:  https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/
* My R package for fraud detection with Benford's Law:  https://github.com/nlinc1905/benfordsLaw
