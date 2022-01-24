# U-net for Melanoma Segmentation

## Purpose

This is a segmentation system to detect the outlines of skin tumors to aid melanoma detection. This system is based on U-Net, which is an arhictecture proposed by Olaf Ronneberger, Philipp Fischer, and Thomas Box in 2015: https://arxiv.org/abs/1505.04597

## Functionality & Usage

### Running the segmentation system

The model will work by simply running the “segmentation_model.py” file in the terminal (assuming all the necessary dependencies specified in “requirements.txt” are installed). By running the file the model will start training using the number of epochs specified in the main function. In every validation  round, the model will write/overwrite image files into the “Preds” folder. These images contain: an actual image, the target layer, and the model’s prediction. 

### Setting up the dataset

The model should work properly with any image dataset (not just skin images) as long as the following holds:

- All original images are stored in the “training”  folder and are in JPG format
- All target masks are stored in the “masks” folder and are in PNG format
- The original images’ file index within the folder matches that of the masks


### File Structure & File Purpose

The model entire model resides within one folder. Inside this folder all the .py files are used for the model itself, while “requirements.txt” simply establishes the package dependencies needed to run this project. The files work as follows:

- “clean.py” contains a simple script we use to clean unnecessary files from the training data
- “unet.py” establishes the entire architecture for the model
- “dataset.py” allows Torch to find the data and match an image with the target mask
- “data_prep.py” performs transforms on the images while also turning them into Torch-friendly tensors
- “segmentation_model.py” brings all these pieces together and runs the entire model
