# Pneumonia diagnosis software
### Pneumonia classifier aimed to automate the process of detecting pneumonia from patients.
![demo](https://user-images.githubusercontent.com/43356500/65567800-7f0f3d80-df25-11e9-923f-d99515b9bc8d.png)

## Getting started

To download the data, simply replace 'xxx' in your Kaggle API ID and Key. You can access this by 
```
My Account -> Create New API Token -> Access Kaggle.json file for ID and Key
```
#### Testing your own images
If you happen to have your chest x-ray picture lying around, or just want to test the model on another chest x-ray picture, simply replace the images in ".chest-xray/val".

## About the model
It uses transfer learning on top of the VGG-16 pretrained weights, with three fully-connected lyaers with dropout regularization and maxpooling. Please refer to this paper for more information on VGG-16. (https://arxiv.org/abs/1409.1556)
![vgg16](https://user-images.githubusercontent.com/43356500/65568155-b9c5a580-df26-11e9-90ba-f41a24e90613.png)

The trained model weights are available in "models" folder. The weights were taken from three different intervals of epochs, at around 35~38 epochs.

## Results
![demo1](https://user-images.githubusercontent.com/43356500/65568006-4459d500-df26-11e9-88ed-2d50c02b382c.png)
![demo2](https://user-images.githubusercontent.com/43356500/65568031-576ca500-df26-11e9-952b-48e2c38feb9a.png)

## Author
* **Richie Youm**

## Disclosure
Please note that this is a side project, and is not meant to replace your doctor's appointment. Seek professional medical help if you feel sick.
