# DeepfakeForensics-v1

This repository holds the code base for basic deepfake detection research. This includes data extraction and augmentation, model training, as well as predicting any 
unseen file or set of files. Four different models are available here by default:

### Image-based models
1. MesoNet4 (https://arxiv.org/abs/1809.00888)
2. EfficientNet (https://arxiv.org/abs/1905.11946)

The image-based models classify each frame individually, and aggregate the frame-level scores by averaging towards a file-level prediction.

### Video-based models
3. MesoNet4 + LSTM
4. EfficientNet + LSTM

Those models are build using the backbones of models 1 and 2, but aggregate the frame-level feaures over a temporal period of N frames. Hence they consider frame-windows of length
N. Again, the file-level prediction is derived by averaging all of the frame-window predictions for the specific file.

## Setting up
Clone repository and set up environment by:

<pre><code>$ pip install -r requirements.txt
</code></pre>

If the environment is not automatically available within Jupyter Notebook, please run

<pre><code>$ python -m ipykernel install --user --name deepfakeforensics --display-name "Python DeepfakeForensics"
</code></pre>

to enable the option to select the environment kernel within the notebook under "Kernel".
Created on Python 3.7 Win 10.

## Data
Due to data non-distribution agreement, no data is available in this repository. Current deepfake detection datasets are:

1. Celeb-DF (can be downloaded via https://github.com/danmohaha/celeb-deepfakeforensics)
2. DeeperForensics-1.0 (can be downloaded via https://github.com/EndlessSora/DeeperForensics-1.0/tree/master/dataset)
3. FaceForensics ++ (can be downloaded via https://github.com/ondyari/FaceForensics)

Once the data is downloaded, follow the steps in the data_extraction notebook. This will create two new folders holding the Labels and extracted data.
For the image- and video-based models, different types of datasets need to be created. Note: the video-based dataset stores the datapoints as pickled tensors, which significantly
increases the required memory compared to image-based data (stored as JPG). 

During this step, we also perform a file-level split into train/val/test sets.

## Training

Next, model can be trained using the dataset. ATTENTION: Different versions of the dataset need to be created for image- and video-based models. Make sure to carefully select the correct
option during data extraction (default is image-based, non temporal data).

The train notebook lets you select any model and train it. Each epoch, a checkpoint is saved to ./checkpoints/\<model name\>/.

## Inference

Once the model is trained, it can be used to predict any collection of video files. Currently however, the setup does not allow for discrimination between multiple faces
in one frame, hence prediction will be faulty if multiple faces are detected in the frame. This will be adressed in upcoming changes.
