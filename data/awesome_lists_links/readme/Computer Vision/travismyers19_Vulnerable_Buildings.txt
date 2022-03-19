# Vulnerable Building Detection
A soft-story building is a multiple-story building in which the first floor is "soft", meaning it consists mostly of garage door openings or windows or other openings.  Soft-story buildings are particularly vulnerable to earthquakes, and thus the ability to quickly inventory soft-story buildings in an area may be of use to city or state officials.  This repo contains code to leverage transfer learning to create and train a custom Inception V3 image classifier to detect soft-story buildings and deploy the model in a user-friendly Streamlit app.

## Overview
The Streamlit app takes as input from the user latitude and longitude coordinates specifying a bounding box region of the world and also a number of addresses.  The app generates random latitude and longitude locations within that bounding box and uses reverse geocoding to obtain addresses.  Then it uses those addresses to obtain Google Street View images.  The images are then sent through the trained image classifier, the bad images are discarded, and the results are presented to the user: "Soft-Story" means that the image is of a soft-story building and "Non-Soft-Story" means that the image is of a non-soft-story buildng.

There are two types of models that can be used:

- Ternary classification:  A single model classifies the images into the three categories:  "Soft", "Non-Soft", and "Bad Image" (a bad image is one in which there is no building in the image or the building is obscured or it's unclear which building the image is an image of).

- Binary classification:  One model classifies the image as good or bad, and then a second model classifies the image as soft or non-soft.

## Installation
Clone the Github repo:

```
git clone https://github.com/travismyers19/Vulnerable_Buildings
```

Change current directory to the project directory:

```
cd Vulnerable_Buildings
```

All command line commands below assume that the user is in the project directory unless otherwise specified.

## Initial Setup

### If using the Ubuntu Deep Learning AMI:

Activate the tensorflow 2.0 with Python 3.6 environment:

```
source activate tensorflow2_p36
```

Install Streamlit:

```
pip install streamlit
```

### If not using the Ubuntu Deep Learning AMI, but using conda:
Create a conda environment from "configs/environment.yml:

```
conda env create -f Configs/environment.yml
```

Activate the conda environment:

```
source activate tensorflow2_p36
```

### If using neither conda nor the Ubuntu Deep Learning AMI:
Install from `requirements.txt`:

```
pip install -r requirements.txt
```

## Modules
The `Modules` folder contains all of the custom modules in this repo.

### `addresses.py`
This module contains the class Addresses which provides functionality for grabbing random addresses using reverse geocoding and for getting images from Google Street View corresponding to given addresses.

### `buildingclassifier.py`
This module contains the class BuildingClassifier which provides functionality for creating and training custom Inception V3 models for either ternary classification or binary classification, as well as functions for evaluating a model to determine the following statistics given a directory containing test images:

- Accuracy:  Percentage of the test images labeled correctly.
- Soft Precision:  Precision in determining soft vs. non-soft.
- Soft Recall:  Recall in determining soft vs. non-soft.
- Good Precision:  Precision in determining good image vs. bad image.
- Good Recall:  Recall in determining good image vs. bad image.

### `customlosses.py`
This module contains custom loss functions for binary crossentropy and categorical cross entropy which incorporate the focal loss modification described in this paper:  https://arxiv.org/abs/1708.02002

### `imagefunctions.py`
This module contains functions for saving a single image and for loading a single image into a numpy array that can be fed to the model for prediction.

## Collecting Data
The folder `Small_Data` contains a small amount of data that can be used.  The following scripts provide functionality for collecting more data and, by default, saving it in the `Data` folder.  Each of these scripts requires a Google API Key.  By default, it is assumed that the api key is located in the project directory in a text file called `api_key.txt`.

### `get_random_addressees.py`
Gets random addresses from a region specified by latitude and longitude coordinates and writes them to a csv file.  A file with 100 such random addresses is provided in `Addresses/random_addresses.csv`.

Type `python get_random_addresses.py -h` in the command line to view the list of arguments that can be passed to this script.

### `get_soft_story_images.py`
Gets Google Street View images given a csv file of addresses and saves them to the `Data` folder by default.  A list of soft-story addresses provided by the city of San Francisco is located at `Addresses/Soft-Story-Properties.csv`.

Type `python get_soft_story_images.py -h` in the command line to view the list of arguments that can be passed to this script.

### `get_non_soft_story_images.py`
Given a csv file of addresses, displays the Google Street View image of each one for the user to manually label as "non-soft-story" (press "y' when the image appears) or "bad image" (press "u" when the image appears).  By default it saves the manually labeled images to the `Data` folder.

Type `python get_non_soft_story_images.py -h` in the command line to view the list of arguments that can be passed to this script.

## Creating a Custom Inception Model Using Transfer Learning
Run `create_inception_model.py` in the command line to create and save a custom Inception V3 model:

```
python create_inception_model.py
```

Type `python create_inception_model.py -h` in the command line to view the arguments that can be passed to the script:
```
  --model_filename MODEL_FILENAME
                        The location to save the created model. Default is
                        'Models/model.h5'.
  --number_categories NUMBER_CATEGORIES
                        The number of output categories for the model (3 =
                        ternary classifier, 1 = binary classifier). Default is
                        3.
  --dense_layer_sizes DENSE_LAYER_SIZES
                        A list of the sizes of the dense layers to be added
                        onto the pretrained model. Default is '[1024, 512,
                        256]'.
  --dropout_fraction DROPOUT_FRACTION
                        The dropout fraction to be used after each dense
                        layer. Default is 0.2
  --unfrozen_layers UNFROZEN_LAYERS
                        The number of layers to unfreeze for training. Default
                        is 21.
  --focal_loss FOCAL_LOSS
                        Modify loss function to use focal loss. Default is
                        False.
```
## Training the Model
Run `train.sh` to train a model:

```
./train.sh
```

The training is run using a bash script in order to utilize the Horovod package which allows for distributed computing.
Set the following variables within the `train.sh` bash script:

- `HOSTS`:  if you want to use distributed computing, list all hosts that will be used.  If only one host is listed, it doesn't matter what that host is because only the localhost will be used.
- `GPUS`:  the number of GPUs on each host.  Default: `1`.
- `TRAINING_DIRECTORY`:  the directory where the training images are located.  If using binary classification, there should be two subfolders in this directory; if using ternary classification, there should be three subfolders.  Default: `Small_Data`.
- `TEST_DIRECTORY`:  the directory where the test images are located (for the purpose of calculating loss and accuracy).  Default: `Small_Data`.
- `MODEL_FILENAME`:  the location of the model to be trained.  Default:  `Models/model.h5`.
- `TRAINED_MODEL_FILENAME`:  the location to save the trained model.  Default:  `Models/trained_model.h5`.
- `METRICS_FILENAME`:  the location to save the loss and accuracy.  It will be saved as a numpy array where the first row is the accuracy in each epoch and the second row is the loss in each epoch.  Default:  `metrics.npy`.
- `WEIGHTS`:  the weights to apply to each class to combat class imbalance.  Default:  `"None"`.
- `BINARY`:  set to 0 if ternary classification, set to 1 if binary classification.  Default: `0`.
- `EPOCHS`:  the number of epochs to train.  Default:  `5`.

## Plotting Training Metrics Using Streamlit
Run `plot_metrics.py` in the command line:

```
streamlit run plot_metrics.py
```

To specify the location of the metrics file, use the argument `--metrics_filename`:

```
streamlit run plot_metrics.py -- --metrics_filename "Models/metrics.npy"
```

## Launching the Streamlit App
Run `product.py` in the command line:

```
streamlit run product.py
```

This will output an external http address that any browser can view.
Type `python product.py -h` n the command line to view the arguments to pass to the script:

```
  --api_key_filename API_KEY_FILENAME
                        The file location of a text file containing a Google
                        API Key. Default is 'api-key.txt'.
  --model_filename MODEL_FILENAME
                        The file location of the model to serve. Default is
                        'Models/trained_test_model.h5'.
  --model2_filename MODEL2_FILENAME
                        If using a binary classifier, specify the file
                        location of the second model. If using a ternary
                        classifier, set to 'None'. Default is 'None'.
```

For instance, to specify the `api_key_filename` argument:

```
streamlit run product.py -- --api_key_filename "api-key.txt"
```