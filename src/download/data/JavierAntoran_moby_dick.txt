# moby-dick Whale Detection

This repo contains our approach to to solving the
[Kaggle Cornell Whale detection challenge](https://www.kaggle.com/c/whale-detection-challenge). The task
is to identify North Atlantic Right Whale (NARW) upcalls from a dataset of audio recordings. These recordings
are taken by microphones placed on buoys.
This makes the task difficult as sea movements introduce a lot of noise.
Also, NARWs and other marine mammals have calls which are hard to distinguish.

[Here](https://vimeo.com/227009627) is a video clip showing how a NARW upcall sounds.

Some sample spectrograms:\
<img src="images/spectrogram_comparison.png" width="680" height="350" />

We try three approaches for classification:
HMMs, Neural Nets, and Gradient Boosting. We show a different
 spectrogram image processing scheme for each approach.

We are provided with 30000 audio clips of 2 second duration, sampled at 2kHz.
This gives us frequency resolution up to 1kHz. NARW upcalls are generally in the range of 50-450 Hz.
We therefore apply a downsampling factor of 2 to all audio clips.
 The dataset is unbalanced with only 7027 positive examples (23.4%).
 There are also 70000 test clips which are unlabelled.
 We do not use these as we don't have the labels, we
 will use K fold cross validation instead.

[Results and Evaluation](#results-and-Evaluation)\
[Preparing the dataset](#preparing-the-dataset)\
[Neural network approach](#neural-networks)\
[HMM approach](#Hidden-markov-models)\
[Gradient boosting approach](#gradient-boosting)\
[Additional resources](#additional-resources)

For a more in depth overview see the project <a href="slides/whale_presentation.pdf" download>slides</a>.

## Results and Evaluation

We achieve a best mean result of 0.9786 roc-auc after with a 0.0014 standard
deviation when running 10 fold cross validation using 10% of data for testing.
This result could probably be improved by ensembling the neural net with
the gradient boosted trees. Another untested option is to give the neural network
some of the features used with the gradient booster as additional inputs.

A summary of our results:


| ROC-AUC | CNN 25ms | CNN 250ms |   HMM  | Grad Boosting |
|:-------:|:--------:|:---------:|:------:|:-------------:|
|   **mean**  |  0.9656  |   0.9786  | 0.6101 |     0.9347    |
|   **std**   |  0.0045  |   0.0014  | 0.0605 |     0.004     |

ROC curves obtained with our best performing method (Neural Net with wide FFT window).

<img src="images/best_ROC_250.png" width="360" height="290" />

## Preparing the dataset

First download the dataset and extract the whale_data folder.
Run the following script in order to save train samples and labels to .npy files.
```bash
python read_data.py
```
This should generate two files: whale_traindata.npy and whale_trainlabels.npy. These will
be used as input to our feature extraction scripts.

## Neural Networks

We represent sound clips as spectrograms, apply some basic image processing
and feed them into our classifier CNN.

### Feature extraction

[Notebook for this section](https://github.com/JavierAntoran/moby_dick/blob/master/Notebooks/format_data_NN.ipynb)

First, the time series are separated into overlapping frames. We choose a frame
size of 25ms and a time advance between frame starts of 10ms. These are
typical values for speech processing. However, because of the time-frequency uncertainty
principle, <img src="https://latex.codecogs.com/gif.latex?\Delta\.t \Delta\.f \geq \frac{1}{4\pi}  " /> , with a small window, our estimation of frequency coefficient
will have more variability. This is especially true for lower frequencies,
where whale upcalls reside. For this reason, we generate a second set of features
with a 250 ms frame and a 11 ms step between frame starts (this values is chosen in order to
obtain a roughly similar amount of frames in both cases).

We apply a hamming window and calculate the 256 point fft of every frame.
We keep the first 128 coefficients as our data is real and the rest of the
coefficients will be symmetric. The resulting spectrograms are shown in the
following plot:

<img src="images/fft_250_25ms.png" width="700" height="290" />


We generate a filter bank of 32 filters taken the region of 40-1000 Hz. In this range,
 the MFB weighs all frequencies almost linearly, with slightly more resolution
 assigned to the lower frequencies.
 This is consistent with the energy distribution of whale upcalls.
 We choose 32 mel filter bands as maximum size of the receptive fields for individual units
 of our network will be of size 32x32. The whale filter bank matrix is shown in the following image:

 <img src="images/whale_filterbank.png" width="360" height="290" />

Finally, we compute the first and second derivatives of our features with respect to time (delta features).
We stack the spectrogram and its derivativ es as to form channels which we will
pass as input to our CNN.

<img src="images/delta_feats.png" width="400" height="500" />


To parse the data with a 25ms window run the following commands:
```bash
cd NN_solution
mkdir data
cp ../whale_traindata.npy data
cp ../whale_trainlabels.npy data
python format_data_25ms.py
```

To parse the data with a 250ms window run the following commands:
```bash
cd NN_solution
mkdir data
cp ../whale_traindata.npy data
cp ../whale_trainlabels.npy data
python format_data_250ms.py
```

Extracted features will be saved to the /NN_solution/data directory.

### Network architecture, Training and Evaluation

We use a 9 layer fully convolutional network, a slightly modified version of the simplenet v2 architecture: https://github.com/Coderx7/SimpNet.
Our implementation, which is heavily based on [this one](https://github.com/Coderx7/SimpleNet_Pytorch),
 is contained in the NN_solution/delta_spectrogram_simplenet/simplenet.py file. We use the same architecture for features
obtained with 25ms and 250ms fft windows. The corresponding input feature sizes
are Nx3x198x32 and Nx3x196x32. We use random cropping on the 25ms features to
get a Nx3x196x32 shaped input.

We use a 90/10% train/validation data split with 10 fold cross validation.
The resulting ROCs are shown in the following plots:

<img src="images/NN_results.png" width="650" height="290" />

25ms window, no cross validation
```bash
cd NN_solution/delta_spectrogram_simplenet/
python run1_train.py
```
25ms window, cross validation
```bash
cd NN_solution/delta_spectrogram_simplenet/
python run2_cross_validate.py
```
250ms window, no cross validation
```bash
cd NN_solution/delta_spectrogram_simplenet/
python run3_train_widewindow.py
```
250ms window, cross validation
```bash
cd NN_solution/delta_spectrogram_simplenet/
python run4_cross_validate_widewindow.py
```

Results will be saved to the NN_solution/results directory.
Pytorch models will be saved to the NN_solution/models directory.

## Hidden Markov Models

We separate our data into windows, we extract 30 multiresolution features from each window
using the stationary wavelet transform. We then use them to train 2 GMM-HMMs.
 We train one on positive examples and the other one
on negative examples. We calculate the probability of there being a whale call
as the softmax of the normalized log likelihood of
a sequence under each HMM.

### Feature extraction

We use the stationary transform (SWT) to get around the time-frequency resolution
trade off while maintaining the length of our time series. This is important as we
need to be able to temporarily align the SWT coefficients extracted at each level.

The following image shows how the wavelet transform applies a differently sized
window to each frequency band.\
<img src="images/wavelet_resolution.png" width="400" height="220"/>

Instead of downsampling the signal as is done in the DWT, the SWT upsamples the filter after each step.
This is more computationally expensive but preserves signal length. The following schematic shows the SWT
filtering chain.

<img src="images/swt_schematic.png" width="600" height="220"/>

We use the db2 wavelet which has the following form:

<img src="images/db2.png" width="300" height="200"/>

We keep the approximation signals (output of low pass filters) at levels 1, 2 and 3.
We then separate these signals into 250ms frames with a 10ms step between frame starts.
We apply a hamming window to each frame and compute their 256 point FFT. Again, because
our original signal is real valued, we obtain 128 frequency coefficients.

<img src="images/swt_spectrograms.png" width="700" height="200"/>

We then apply our whale filter bank to all three signals. We only keep coefficients
that are located in the passing band of their corresponding wavelet. We stack these
coefficients in order to form the following multiresolution feature map:

<img src="images/multiresolution.png" width="300" height="250"/>

We then apply a DCT transformation to each frame's frequency coefficients in order
to obtain decorrelated features. Finally we normalize the features by
 subtracting their the mean and diving them by their standard deviation.
 A regular spectrogram and its corresponding feature map are shown
side by side:

<img src="images/dct_multiresolution.png" width="630" height="280"/>

Run the following commands in order to execute the described feature extraction
procedure.
```bash
cd HMM_solution/
mkdir data
cp ../whale_traindata.npy data
cp ../whale_trainlabels.npy data
python format_data_wavelets_dct.py
```

Extracted features will be saved to the /HMM_solution/data directory.

### HMM Training and Evaluation
Our HMM implementation is written in numpy and can be found in the HMM_solution/hmm/modules/HMM.py file. We train
them using the Viterbi algorithm. We place a Dirichlet prior on state transition
probabilities in order to aid training. This also assures that we end up with no
0 probability transitions.
Our GMM implementation is also written in numpy and can be found in the HMM_solution/hmm/modules/GMM.py file.
We train our GMMs with two EM algorithm steps each time we pass our data through
the HMM.

<img src="images/HMM_GMM.png" width="380" height="220"/>

We obtain poor results with this method. Using a 10 hidden state HMMs with
a 3 component GMM per hidden state, we obtain the following ROC:

<img src="images/best_hmm_roc.jpg" width="360" height="290"/>

To train both HMMs, run the following commands:
```bash
cd HMM_solution/hmm
python run1_train_hmm.py
```

To evaluate the HMMs and generate the ROC curve, run:
```bash
cd HMM_solution/hmm
python run2_eval_hmm.py
```

Results will be saved to the HMM_solution/results directory.
HMM models are saved as pickle files to the HMM_solution directory.

## Templates and Gradient Boosting


### Feature extraction

We generate 300 templates composed of lines of different lengths and inclinations
in order to match the shape of whale upcalls. Calculate the cross correlation
 of these templates with our spectrograms in order to probe for different time-frequency
 shift rates in the data. This technique is a proof of concept. Similar results can
 be obtained in a more efficient manner by estimating gradients for individual
 directions (x, y) in the image and then combining them at each point. Additionally,
 higher order, curvature information could also be used to improve results.

<img src="images/template_xcorr.png" width="660" height="350"/>

Generate spectrograms:
```bash
cd /template_boosting_solution
mkdir data
cp ../whale_traindata.npy data
cp ../whale_trainlabels.npy data
python format_data.py # Get spectrograms
python paraidiots_data.py 10
```

With 30000 spectrograms and 300 templates, processing all of our data will
require 9000000 2d correlations. This will be slow on a single CPU, and is best
run as a distributed task. 'python paraidiots_data.py 10' creates 3000 chunks of 10 spectrogams for task distribution.
We extract the following features from each cross correlated feature map:
max value, mean value, std all values. We also extract the following axis wise (x, y) features:
mean, std, skewness and kurtosis. This results in 3300 features per spectrogram.

In order to achieve task distribution, we make use of HTCondor(https://research.cs.wisc.edu/htcondor/) which was installed on the server. It is a software framework for coarse-grained distributed parallelization of computationally intensive tasks.
Through a custom bash script we are able to submit N tasks to condor queue. In this case there will be 3000 tasks computing the spectrogram features.

```bash
python generate_templates.py 3000
```

### Model Training and Evaluation

We use the extracted features as inputs to a gradient boosting algorithm: [XGboost](https://xgboost.ai).
We obtain decent results, especially considering the simplicity of our feature extraction
approach:

<img src="images/best_ROC_xcorr.png" width="360" height="300"/>

Run without cross validation:
```bash
cd template_boosting_solution
python gradient_booster.py
```

Run, cross validation:
```bash
cd template_boosting_solution
python gradient_booster_crossvalidate.py
```

## Additional resources

An assortment of publications related to Neural Nets, HMMs, Gradient Boosting and Marine
Mammal audio detection can be found
in the [papers](https://github.com/JavierAntoran/moby_dick/tree/master/papers) folder.

* Contest winners repo: https://github.com/nmkridler/moby2
* Simplenet paper: HasanPour, Seyyed Hossein et al. “Let ’ s keep it simple : Using simple architectures to outperform deeper architectures.” (2016).\
https://arxiv.org/pdf/1608.06037.pdf
* XGboost: [website](https://xgboost.ai)\
 Tianqi Chen and Carlos Guestrin. 2016. XGBoost: A Scalable Tree Boosting System. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '16). ACM, New York, NY, USA, 785-794. DOI: https://doi.org/10.1145/2939672.2939785

