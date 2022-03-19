# Interpolating high granularity solar generation and load consumption power data using SRGAN
The code implementation for interpolating high granularity solar generation and load consumption power data using super resolution generative adversarial network (SRGAN). This work has been published on Applied Energy: http://dx.doi.org/10.1016/j.apenergy.2021.117297.
The original SRGAN work can be found here: https://arxiv.org/abs/1609.04802.
## Introduction
To date, most open access public smart meter datasets are still at 30-minute or hourly temporal resolution. While this level of granularity could be sufficient for billing or deriving aggregated generation or consumption patterns, it may not fully capture the weather transients or consumption spikes. One potential solution is to synthetically interpolate high resolution data from commonly accessible lower resolution data, for this work, the SRGAN model is used for this purpose.
## Requirements
* Python 2.7.13 
* tensorflow==1.9.0
* Keras==2.2.4
* numpy==1.15.2
* pandas==0.23.4
## Datasets
The pretrained models are trained using one year of PV & consumption power data of 2340 Australian households, collected by Solar Analytics (https://www.solaranalytics.com/au/) via Wattwatcher energy monitors (https://wattwatchers.com.au/).
## Files
* pre_trained_models/: directory for the four pre-trained models which interpolate PV & load from 30-minute & hourly measurements.
* datasets.py: functions to load training/evaluation datasets.
* networks.py: model architectures for the generator and discriminator.
* SRGAN_train.py: script to train a SRGAN model.
* SRGAN_test.py: script to generate 5-minute PV or load data using a trained SRGAN model.
## Code References
The codes are built upon the SRGAN implementation from https://github.com/deepak112/Keras-SRGAN and the DCGAN implementation from https://github.com/eriklindernoren/Keras-GAN.
