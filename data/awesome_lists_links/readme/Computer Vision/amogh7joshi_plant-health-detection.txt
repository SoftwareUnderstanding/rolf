# Plant Disease Identification

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/amogh7joshi/plant-health-detection/blob/master/model.ipynb)

**Note**: **This was one of the first projects I developed when working with deep learning, and it is no longer monitored
or in use. You can work with it on your own, but no new changes will be made, since this project is complete.** 

Identifying the health of plants is a lengthy but necessary process in order to keep plants 
healthy. The issue is especially intensified in regions with expansive farming land and crop growth.
Rather than manual identification, neural networks can be used to identify plants that are healthy or diseased.

This repository contains the source code for neural networks which are used to identify plant diseases. 

## Installation

To install this repository, you can directly clone it from the command line:

```shell script
git clone https://github.com/amogh7joshi/plant-health-detection.git
```

Then, enter the repository and install system requirements.

```shell script
# Enter the Repository
cd plant-health-detection

# Install System Requirements
python3 -m pip install -r requirements.txt
```

Then, you can download the plant leaf health dataset from [this location](https://data.mendeley.com/datasets/hb74ynkjcn/1). It will take up
around 6 GB of space. Move the folder containing the dataset into the `data` subdirectory. Then, run the `preprocess.sh` script in order to preprocess the data.
It takes around 3 minutes to process each of `healthy|diseased` images, so the script splits them up if you  would like.

## Model Information

The model that I am currently using is based roughly off of ResNet [\[1\]](http://arxiv.org/abs/1512.03385), and manages to achieve an impressive 99% accuracy on training 
data at 100 epochs. The link at the top contains the model training script in Google Colab. 

## References

\[1\]: He, K., Zhang, X., Ren, S., & Sun, J. (2015). Deep residual learning for image recognition. ArXiv:1512.03385 [Cs]. http://arxiv.org/abs/1512.03385

