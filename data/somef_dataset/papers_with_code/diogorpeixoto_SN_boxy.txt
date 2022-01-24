# SN_boxy

The goal of this project was to perform vehicle detection across different types of weather, traffic and light conditions. 
To do so I used an implementation of YOLOv3 (https://github.com/experiencor/keras-yolo3) with Switchable Normalization (https://arxiv.org/abs/1806.10779 , currently in development) and with Batch Normalization on the boxy dataset (https://boxy-dataset.com/boxy/).

Slides of the latest demo can be found here: http://bit.ly/driveflexMLslidedeck

## This repository is organized as follows:

- **utils** : All source code for production
- **tests** : All code for testing
- **configs** : Configuration files.
- **data** : Small amount of example data from boxy dataset to validate installation
- **deps** : Dependencies with code from YOLO v3 repository

## Setup

Clone repository and create a new environment
```
conda create -n SN_boxy python=3.6
source activate SN_boxy
cd SN_boxy
pip install -r requirements.txt
```

### Image Data

For simplicity and to allow for bash scripts to download the data all the urls are listed in file: boxy_file_list_all.txt
Place training and validation data in data directory. 
Full boxy dataset available at https://boxy-dataset.com/boxy/.


### Annotation Data

Original annotation data for train and validation in the boxy dataset is available in 'boxy_labels_train.json' and 'boxy_labels_valid.json' files whose urls are in boxy_file_list_all.txt


### Test Annotation Data

- Test that json labels from boxy dataset (training and validation) are valid
```
# Example

python label_checks.py -/SN_boxy/labels_train/boxy_labels_train.json 

```

### Convert Annotation Data

To allow YOLO to use annotations in VOC format run jsonToVOC2.py in the 'utils' directory as so:

```
python jsonToVOC2.py <json file path> <output directory name>  <resize factor>

# Example

python jsonToVOC2.py -/SN_boxy/labels_train/boxy_labels_train.json -/SN_boxy/labels/train  2

```

Please keep in mind that if using the full resolution dataset from boxy you should use 'resize factor' = 1.
If using the scaled down version of the images you should use 'resize factor' = 2.  


### Configs


- Use the available config files in /SN_boxy/configs/

Edit files such that, "train_image_folder" , "train_annot_folder" , "valid_image_folder" , "valid_annot_folder" are set to the directories where the training images and annotations (after conversion to VOC format) are stored. Same hold for the validation images and annotations.

## Usage 

### Run Inference

- Run predict.py on raw images to detect vehicles. Use the default config_boxy.json file 
```
# Example

python predict.py -c /SN_boxy/configs/config_boxy.json -i /SN_boxy/data/raw/

```

### Evaluate Model

- Run evaluate.py on raw images to detect vehicles and compare performance to ground truth from annotation files. Ensure that configuration file (config_boxy_evaluate.json) is set up correctly according to the Configs instructions above.

```
# Example

python evaluate.py -c /SN_boxy/configs/config_boxy_evaluate.json

```

### Train Model

- Run train.py on images and corresponding annotation files to learn to detect vehicles. Ensure that configuration file (config_boxy_evaluate.json) is set up correctly according to the Configs instructions above.

```
# Example

python train.py -c /SN_boxy/configs/config_boxy_train.json

```


