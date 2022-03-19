# Contamination_DenseNet
Contamination classification for explants

This repository contains the pipeline for classifying explants into categories - contaminated, non-contaminated, missing

This is [Keras](https://keras.io/) implementation of DenseNet. The code for densenet used in this repository is obtained from [here](https://github.com/flyyufelix/cnn_finetune).

To know more about how DenseNet works, please refer to the [original paper](https://arxiv.org/abs/1608.06993)

```
Densely Connected Convolutional Networks
Gao Huang, Zhuang Liu, Kilian Q. Weinberger, Laurens van der Maaten
arXiv:1608.06993
```

### Setup environment.

1. Create a conda/python virtual environment.

2. Install the dependencies from requirements.txt using pip -

```
    conda create --name <envname> --file requirements.txt
```

To manually install dependencies, follow the below steps - 

1. Create a conda create environment: 

```
conda create -n envname  python=3.6
conda activate test1
```

2. Install Keras and Tensorflow-gpu using Conda

```
conda install -c conda-forge keras
conda config --set restore_free_channel true
conda install tensorflow-gpu=1.13
```

4. Pip install libraries

```
pip3 install opencv-python Pillow matplotlib easydict argparse tqdm
```

Using Conda 

```
conda install -c conda-forge opencv matplotlib easydict argparse tqdm
conda install -c anaconda pillow
```


### To run predictions on the dataset, please follow the below steps.

1. Create a directory for the input rgb images, and place the dataset into this directory.

   Please note that currently the code only supports grid type 12 (3 X 4).

2. Please place the downloaded model in the main working directory.

3. Create a CSV which contains the list of input RGB images. Check test.csv in the folder.

3. To run the script, use the command -

Please update the output directory path in the config.py file (OUTPUT)

```
    KERAS_BACKEND=tensorflow python inference.py --img-list test.csv 
```

Or, Include img-list, crop_dims and output_file paths as arguments to the script.

```
    KERAS_BACKEND=tensorflow python inference.py --img-list test.csv --output_file output_file_name --crop_dims "(260, 600, 1700, 1710)"  
```

To turn on debugging, use the debug flag and set it to True. (--debug True)

Format of the output CSV - 

```
image_name,1,2,3,4,5,6,7,8,9,10,11,12
GWZ7_I2.0_F1.9_L80_194153_8_1_3_rgb.jpg,NC,NC,NC,NC,NC,NC,M,NC,NC,NC,NC,NC
GWZ7_I2.0_F1.9_L80_194927_1_2_5_rgb.jpg,C,NC,NC,NC,NC,NC,NC,NC,NC,NC,C,NC
GWZ7_I2.0_F1.9_L80_194429_11_2_2_rgb.jpg,NC,NC,NC,NC,NC,C,NC,NC,NC,NC,NC,NC
GWZ8_I2.0_F1.9_L80_195913_4_0_6_rgb.jpg,C,NC,NC,C,C,C,C,NC,C,C,C,NC
```

## Requirements

* Keras 2.0.5
