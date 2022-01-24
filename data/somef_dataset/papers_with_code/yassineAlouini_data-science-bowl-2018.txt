# Data Science Bowl 2018


This project is a tool for detecting cells in medical images.
It is based on the following [Kaggle challenge](https://www.kaggle.com/c/data-science-bowl-2018 ).


## Project structure


TODO: Add structure

## Install the project

For this, run the following command:

`conda install --file environement.txt`


Then, run:

`python setup.py develop`

## Workflow

1. Get and process the data
2. Train the model
3. Postprocess the predictions



## Get the data


Once you have installed the project dependencies, you will have access to the Kaggle official CLI. Run `kaggle --help` to confirm this.


To get the data, run:

`kaggle competitions download -c data-science-bowl-2018`

You will need to accept the competition conditions and create an API key first.

## Running TensorBoard

To run TensorBoard (a great visualization tool),

tensorboard --logdir=/path/to/tb_logs

## Tips

* It is better to use skimage instead of numpy for reading and processing images.
* It is even better to use Keras built-in image processing capabilities.
* Log the various image sizes when debugging your data processing pipeline.
* Use only one image when debugging your data pipeline (so that to avoid loading all the data multiple times).

## Sources and useful links

* U-net Kaggle kernel: https://www.kaggle.com/keegil/keras-u-net-starter-lb-0-277
* U-net paper: https://arxiv.org/abs/1505.04597
* Upsampling basics: https://www.cs.toronto.edu/~guerzhoy/320/lec/upsampling.pdf
