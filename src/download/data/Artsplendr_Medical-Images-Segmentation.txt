# Medical-Images-Segmentation

## Goal of the Project
Biomedical images projects often have been experiencing a problem of insufficient number of annotated input images available.
Additionally, medical imaging requires localisation (e.g. classification of every pixel) or medical image segmentation.
Goal of this project is to reproduce medical images segmentation experiment (author: Soriba Diaby) and to learn how dataset with few samples could be used for medical images segmentation.

## Dataset
Very small dataset of 3D CT scans will be used in this project:  the dataset contains 3D CT scans of 20 patients (10 women, 10 men) with in total 75 % cases of tumours.  The dataset includes liver mask segmentations as well.
To overcome scarcity of the data, slicing 3D Images into 2D slices is used.
The dataset 3D-IRCADb-01 is available for download in NIfTI format at the following link: <https://www.dropbox.com/s/8h2avwtk8cfzl49/ircad-dataset.zip?dl=0>.


## Model
The classic U-net architecture will be used for the segmentation task, as described in the original paper “U-Net: 
Convolutional Networks for Biomedical Image Segmentation” by Olaf Ronneberger, Philipp Fischer, and Thomas Brox.

## Predictions
Few examples of predictions below:
  
<p align="left">
  <img width="631" height="188" src="images/Medical_Images_Segmentation(Liver).png">
</p>
  
 

## References
1. “Medical images segmentation with Keras: U-net architecture” by Soriba Diaby at this link <https://towardsdatascience.com/medical-images-segmentation-using-keras-7dc3be5a8524> and at the following link: <https://github.com/soribadiaby/Deep-Learning-liver-segmentation>;
3. “U-Net: Convolutional Networks for Biomedical Image Segmentation” research paper by Olaf Ronneberger, Philipp Fischer, and Thomas Brox is available here: <https://arxiv.org/pdf/1505.04597.pdf>;
