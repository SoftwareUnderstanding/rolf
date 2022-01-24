# Spatial-Transformer-Network-application
Application of Spatial transformer network to a hand alignment task.

All results refer to the dataset avaiable <a href="https://drive.google.com/open?id=1Kc66EO2p9rw08ZKu6PknQlPZKJsHiCfo">here</a>.

A full description of this project is provided in <a href="https://github.com/kinik93/Spatial-Transformer-Network-application/blob/master/ImageAnalysisArticle.pdf">ImageAnalysisArticle.pdf</a>

## Getting started
If you want to create your own dataset run [acquisition.cpp](https://github.com/kinik93/Spatial-Transformer-Network-application/blob/master/acquisition.cpp) source file on Windows OS and arrange the dataset folder like we did. 
The [STNtest](https://github.com/kinik93/Spatial-Transformer-Network-application/blob/master/STNtest.ipynb) script includes the code to test the network.

### Dataset creation requirements
The file [acquisition.cpp](https://github.com/kinik93/Spatial-Transformer-Network-application/blob/master/acquisition.cpp) has the following dependecies:
* [DepthSense SDK](https://www.sony-depthsensing.com/Support/DownloadLegacyDriver)
* [OPENCV 2.4.9](https://opencv.org/releases.html)

and obviously a depth camera like *Senz3d* or *DepthSense*.

An example of depth frames (8 bit PNG encoding) acquisition is shown in the following gif:

![Alt Text](https://github.com/kinik93/Spatial-Transformer-Network-application/blob/master/makingOfDataset.gif)

### STN requirements

Package name | Version
------------ | -------------
[Python](https://www.python.org/) | 2.7 
[Numpy](http://www.numpy.org/) | 1.14.3 or higher
[OpenCV](http://opencv-python-tutroals.readthedocs.io/en/latest/) | 3.4.1 or higher
[Tensorflow](https://www.tensorflow.org/) | 1.7.0 or higher
[Keras](https://keras.io/) | 2.1.6 or higher
[matplotlib](https://matplotlib.org/) | 2.2.2 or higher
[PIL](https://pillow.readthedocs.io/en/3.0.x/index.html) | 3.4.2 or higher

and [SpatialTransformer.py](https://github.com/kinik93/Spatial-Transformer-Network-application/blob/master/spatial_transformer.py) [2]

## Run the scripts
The notebook [STNtest.ipynb](https://github.com/kinik93/Spatial-Transformer-Network-application/blob/master/STNtest.ipynb) file is provided for a user friendly mode to test the application. Pretrained model weights can be loaded from [weights_last.h5](https://github.com/kinik93/Spatial-Transformer-Network-application/blob/master/weights_last.h5).

We recommend to use [STN_train.py](https://github.com/kinik93/Spatial-Transformer-Network-application/blob/master/STN_train.py) in a real use context to train the network, just type:

```
python STN_train.py
```

## Experiment results

#### Loss functions
<img src="https://github.com/kinik93/Spatial-Transformer-Network-application/blob/master/lossTrend.png" alt="Loss function">

#### Training process gif sample

![Alt Text](https://github.com/kinik93/Spatial-Transformer-Network-application/blob/master/depth.gif)


## References
 * [1] https://arxiv.org/abs/1506.02025
 * [2] https://github.com/oarriaga/spatial_transformer_networks

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Authors:
* Niccolò Bellaccini, nicco.b.93@gmail.com, https://github.com/kinik93
* Tommaso Aldinucci, tommaso.aldinucci@icloud.com, https://github.com/tom1092


