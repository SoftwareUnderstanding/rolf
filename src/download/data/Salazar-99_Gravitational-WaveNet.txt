# Gravitational-WaveNet
Event detection of binary mergers via gravitational-wave observations at LIGO requires processing very noisy signals to identify physical parameters. This typically involves generating synthetic data in the form of waveform template banks, spectral analysis, fourier analysis, and performing matched-filtering. All of these steps are computationally expensive and require human-generated heuristics and oversight. 

This project is an experiment designed to form a baseline on the performance of deep learning on a noisy time-series classification problem. The architecture used is named Gravitational-WaveNet for its use of dilated causal-convolutions as introduced in the 2016 WaveNet paper (https://arxiv.org/abs/1609.03499).

The model is trained to classify a signal of fixed length as either containing or not containing a gravitational wave signature. The positive class consists of synthetic gravitational waveforms for like-mass binary mergers generated using the PyCBC library with additive Gaussian noise to achieve a specified signal-to-noise ratio. The waveforms are generated with masses equally spaced in the range specified by the user. The negative class is standard normal gaussian noise. 

The structure of the project is as follows:
* `models.py` - Contains the GWN class for the neural network (subclasses keras.Model)
* `data.py` - Contains functions for generating, transforming, and saving synthetic gravitational wave data
* `datasets.py` - A command line tool for generating and saving datasets
* `experiments.py` - A command line tool for running experiments (training an instance of the model on a dataset) and logging results
* `data/` - Folder containing generated datasets in .npy form to avoid regenerating datasest for every experiment
* `experiments/` - Folder containing automatically generated logs of experiments (hyperparameters, dataset used, and metrics)

The workflow is as simple as generating a dataset

`>> python3 datasets.py --snr 10 --batch_size 100 --sample_rate 60 --mass_range 1 100`

And then performing an experiment 

`>> python3 experiments.py --conv_layers 2 --filters 32 32 --kernel_size 4 4 --dilation_rate 2 --gru_cells 32 --epochs 20 --data_path 'path/to/data'`

Early experiments showed a minimum classification accuracy of about 70% at a SNR of about 300. Not surprisingly, with the conventions used in this work (see the paper), this is the regime in which the signal and noise are visually indistinguishable. In smaller and larger regimes the classification accuracy quickly jumps back to about 99%. Nonetheless, it is evident that the network was able to discern some kind of structure in the positive class other than amplitude given that the classification accuracy did not approach 50% when the classes were similar. Future work could include more in depth-experimentation and hyperparameter tuning, analyzing the data which was misclassified by the network, performing experiments with different noise spectra (such as that provided by LIGO), modifying the network architecture, as well as exploring the use of computationally inexpensive preprocessing that could aid accuracy. 
