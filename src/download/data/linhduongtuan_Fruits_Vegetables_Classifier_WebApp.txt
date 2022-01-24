# Fruits_Vegetables_Classifier_WebApp
Recently, we published a paper "Automated fruit recognition using EfficientNet and MixNet", can be found here:  https://doi.org/10.1016/j.compag.2020.105326
or https://www.researchgate.net/publication/339798572_Automated_fruit_recognition_using_EfficientNet_and_MixNet

This is an neural network webapp visualizing the training of the network and testing accuracy ~ 99.5% accuracy.
The neural network uses pretrained EfficientNet_B0 and then trained to classify images of fruits and vegetables.
It is built using Pytorch framework using Python as primary language.
The webapp is built using Flask.

## Dataset used :     
120 Category Fruits and Vegetables Dataset can be found here:     

https://www.kaggle.com/moltean/fruits
or https://github.com/Horea94/Fruit-Images-Dataset
And the original paper of the dataset was introduced by Horea Mureșan and Mihai Oltean (https://www.researchgate.net/publication/321475443_Fruit_recognition_from_images_using_deep_learning)
## Neural Network used : 
EfficientNet family was introduced by a paper from Google Brain at https://arxiv.org/pdf/1905.11946.pdf
And codes for the EfficientNet family were hacked by Ross Wrightman. Thank Ross for his fantastic work to create valuable models for image classification tasks on PyTorch. We can find the codes of Ross here: https://github.com/rwightman/pytorch-image-models and https://github.com/rwightman/gen-efficientnet-pytorch
* You can download the trained weight of EfficientNet_B0 model [here](https://github.com/linhduongtuan/Fruits_Vegetables_Classifier_WebApp/blob/master/release/EfficientNet_B0_SGD.pth)    

       

## Flow:
* to reproduce our experiments in our paper, you can use [EfficientNet_B0_SGD.ipynb] (https://github.com/linhduongtuan/Fruits_Vegetables_Classifier_WebApp/blob/master/EfficientNet_B0_SGD.ipynb)
* Input image is fed and transformed using : [commons.py](https://github.com/linhduongtuan/Fruits_Vegetables_Classifier_WebApp/blob/master/commons.py)     
* Inference is done by : [inference.py](https://github.com/linhduongtuan/Fruits_Vegetables_Classifier_WebApp/blob/master/inference.py) 
* Run on local web: [app.py] (https://github.com/linhduongtuan/Fruits_Vegetables_Classifier_WebApp/blob/master/app.py) 

## Run on Ubuntu and MacOS, but not test on Windows - 
Make sure you have installed Python , Pytorch, Flask and other related packages, refer requirement.txt.

* _First download all the folders and files_     
`git clone https://github.com/linhduongtuan/Fruits_Vegetables_Classifier_WebApp.git`     
* _Then open the command prompt (or powershell) and change the directory to the path where all the files are located._       
`cd Fruits_Vegetable_Classifier_WebApp`      
* _Now run the following commands_ -        

`python app.py`     


This will firstly download the models and then start the local web server.

now go to the local server something like this - http://127.0.0.1:5000/ and see the result and explore.

## How to cite
If you find our work useful for your research, please cite the paper using the following BibTex entry:
### BibTex
```bibtex
@article{duong2020automated,
  title={Automated fruit recognition using EfficientNet and MixNet},
  author={Duong, Linh T and Nguyen, Phuong T and Di Sipio, Claudio and Di Ruscio, Davide},
  journal={Computers and Electronics in Agriculture},
  volume={171},
  pages={105326},
  year={2020},
  publisher={Elsevier}
}
```
### Latest DOI

[![DOI](https://zenodo.org/badge/168799526.svg)](https://doi.org/10.1016/j.compag.2020.105326)

### @creator - Duong Tuan Linh
