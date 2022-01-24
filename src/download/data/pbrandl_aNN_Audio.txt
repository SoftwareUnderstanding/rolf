# aNN_Audio

Modelling a digital twin of an analog harmonic distortion device for audio signals as shown in the image below. <img src="https://render.githubusercontent.com/render/math?math=x, y"> are audio signals, i.e., 16-bit wav files. Based on these files the neural network is trained to minimize prediction errors.
Essentially, the device adds overtones to the signal. The current implementation uses a modified version of WaveNet [1].

<p align="center">
  <img src="https://github.com/pbrandl/aNN_Audio/blob/master/images/concept.png?raw=true" width="50%" height="50%" alt="Conceptual Digital Twin" align="center">
</p>


Due to computational complexity of the project is mostly implemented in as Python notebook in Google Colab (`WaveNet.ipynb`). Due to the large training data set (currently 4 GB) the data is stored in Google Drive. (If interested just contact me.)

## Training Data Generation

Training data <img src="https://render.githubusercontent.com/render/math?math=x"> is generated in `generateTrainSet.py` as a 16-bit wav. The data used for training is randomly cut and concatenated from MedleyDB -- a dataset of multitrack audio for music research [2]. The obtained audio concatenation was then recorded through an analog harmonic distortion device to obtain the target file <img src="https://render.githubusercontent.com/render/math?math=y"> for training. Note, the input <img src="https://render.githubusercontent.com/render/math?math=x"> differs from the target file <img src="https://render.githubusercontent.com/render/math?math=y"> only by the distortion effect from the device. 

## Preprocessing

Note that the input and target files are starting and ending with a loud click sound. This helps for synchronization of the files after recording (recording may start or end with arbitrary offset before or after the actual audio signal). To synchronize the two files are trimmed according to the loud click sound. This results in equal length of both files. This is done in `preprocessing.py`.

## Training the Model and Model Predictions

The model is implemented in a Python notebook in Google Colab (`WaveNet.ipynb`). Input and target files have to be uploaded to Google Drive. Once loaded as tensors of shape (channels, num_samples) in the notebook, the fit function in the notebook can be called and training starts. Once trained, a `predict_sequence` function is implemented, that takes an arbitrary length of audio file as tensor and predicts the harmonic distortion.



<!---## Predicting an Audio Sequence --->

<!--- The WaveNet is constructed to predict an arbitrary length of an audio file. In order to achieve that the input audio file is divided in <img src="https://render.githubusercontent.com/render/math?math=n"> parts. Then, each divison <img src="https://render.githubusercontent.com/render/math?math=x_i \in [x_0, ... x_n]"> is predicted by a forward pass through the model. However, this leads to missing information about the preceeding audio signal <img src="https://render.githubusercontent.com/render/math?math=x_{i-1}">. Therefore, the ending of the signal <img src="https://render.githubusercontent.com/render/math?math=x_{i-1}"> is added to <img src="https://render.githubusercontent.com/render/math?math=x_{i}">. The size of the ending is defined by the receptive field length. ---> 

# Reference
- [1] A. van den Oord, S. Dieleman, H. Zen, K. Simonyan, O. Vinyals, A. Graves, N. Kalchbrenner, A. Senior & K. Kavukcuoglu, "WaveNet: A Generative Model for Raw Audio", in CoRR, 2016, http://arxiv.org/abs/1609.03499.
- [2] R. Bittner, J. Salamon, M. Tierney, M. Mauch, C. Cannam & J. P. Bello, "MedleyDB: A Multitrack Dataset for Annotation-Intensive MIR Research", in 15th International Society for Music Information Retrieval Conference, Taipei, Taiwan, 2014, https://medleydb.weebly.com/.

