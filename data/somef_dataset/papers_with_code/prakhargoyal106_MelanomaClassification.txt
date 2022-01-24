# Using CNN for Melanoma Classification

This project is build for binary classification of Melanoma (Skin Cancer) in images of skin lesions. Melanoma is a deadly disease, but if caught early, most melanomas can be cured with minor surgery. Image analysis tools that automate the diagnosis of melanoma will improve dermatologists' diagnostic accuracy. Better detection of melanoma has the opportunity to positively impact millions of people.

## Built Using

* Pytorch

## Dataset Used

The ISIC 2020 Challenge Dataset https://doi.org/10.34970/2020-ds01 (c) by ISDIS, 2020. Images were then resized to 256X256 size to ease/speed processing. 



## Data Augmentation Technique

* Remove Hair: In the dataset there were many images with body hair covering the lesion area, this could lead our model to learn false information during training. To avoid this a data augmentation technique- RemoveHair was introduced to remove the hairs from the image

![alt text](https://github.com/prakhargoyal106/MelanomaClassification/blob/master/Images/With%20hair.png)

* Microscope: Some of the images in dataset have black areas around the lesion area as if the image was taken from microscope. Using Microscope data augmentation technnique, few more such images were introduced in dataset. This helped increase model accuracy.

![alt text](https://github.com/prakhargoyal106/MelanomaClassification/blob/master/Images/Microscope.png)


## Model

Model used in the  classification was inspired from https://arxiv.org/pdf/1905.11946.pdf. In this paper author systematically study model scaling and identify  that carefully balancing network depth, width, and resolution can lead to better performance. Based on this observation, we propose a new scaling method that uniformly scales all dimensions of
depth/width/resolution using a simple yet highly effective compound coefficient


## Future Work

* Introduce new machine learning optimization techniques to improve model accuracy
* Add more Data Augmentation strategy for better generalization of model
 
