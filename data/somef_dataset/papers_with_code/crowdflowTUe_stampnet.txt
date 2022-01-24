# StampNet Information

Code for ["StampNet: unsupervised multi-class object discovery"](https://arxiv.org/abs/1902.02693) by Visser, Corbetta, Menkovski and Toschi.

## What is StampNet
StampNet is a way for performing multi-class multi-object object discovery. Well, sort of, as it is designed to discover and cluster abstract shapes, not real-world or ImageNet-style images. 

It is easier to see StampNet as a **localising multi k-means clustering** using neural networks. Besides finding *k* clusters by minimising MSE, the network can also:
- localise the clusters (or **stamps** as we call them) simultaneously
- find multiple clusters in each image

Besides the results and applications we show in the paper, StampNet can also be used in other ways. For example:
- StampNet can perform a better clustering of Omniglot compared to, say, *k*-means clustering. If the cluster size are smaller and the network localises the clusters, it makes them more translation invariant resulting in a better clustering. (Not in the paper due to page limit constraint)

The method is fairly robust. The network uses *M* stamps per image, but even when there are less than *M* objects per image, it still manages to reproduce the image properly by learning an empty stamp. When there are more than *M* objects in the image, the network learns to localise the objects with the highest average pixel value. (Not in the paper due to page limit constraint)


## How does StampNet work?
StampNet is an autoencoder with a discrete latent space, e.g. it tries to reproduce the input image. The discrete latent space consists of three [gumbel softmax](https://arxiv.org/abs/1611.01144) that predict the *(x, y)* coordinate and which stamp *s* we should use. We then place the predicted stamp at the predicted coordinate to reproduce the image. For more details, see the [paper]((https://arxiv.org/abs/1902.02693)).


## Installation

### Dependencies

The StampNet code has been written in Python 3 and are dependent on the following packages:

```bash
pip install tensorflow-gpu keras sacred matplotlib scikit-image scikit-learn jupyterlab seaborn munkres opencv-python
```

Dependencies:

- Keras verion: 2.3.1
- Tensorflow: 2.0.0
- Numpy: 1.17.3
- Sacred: 0.8.0

### Datasets

Dataset **CT-MNIST** should be generated before use, which can be done by:

```bash
python generate_cluttered_mnist.py
```

The **Pedestrian** dataset is inside the data folder and needs to be extracted, e.g. 

```bash
tar -zxcf pedestrian.tar.gz
```

## Code explanation

StampNet has been written in Keras using Sacred for experiment production. The parameters for each experiment are stored in their respective python file, e.g. `ct-mnist-2.py`, which calls the main network `stamp-network.py` with the specific parameters. Then:

- The experiment will run
- Experiment data, such as the weights or the parameters, will be stored in the `runs` folder. Each file has their own folder and every run will be stored separately.

The `stampnet_analysis.ipynb` contains a few figures and measures of the paper, which has been tested with Jupyter lab:

```bash
jupyter lab
```

**Experiment 0** contains a sample run for each of the different files (except for T-MNIST-1).
