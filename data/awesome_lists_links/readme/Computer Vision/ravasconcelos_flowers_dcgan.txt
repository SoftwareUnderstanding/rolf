# Flower Image Synthesis – DCGAN Network
Flower generation using Deep Convolutional Generative Adversarial Network

University of Toronto School of Continuing Studies

Term project for Deep Learning course

Group: Ankur Tyagi, Haitham Alamri, Rodolfo de Andrade Vasconcelos

Professor: Sina Jamshidi

## Business Problem
Possible Applications:
* Generate artwork
* Image Synthesis

## Prerequisites
* Python 3.7
* Tensorflow 2
* Jupyter Notebooks

## How to run

### Colab Jupyter Notebook
It is better to run the Notebook on Colab Pro with GPU and High-RAM.
1. Open
1. Run all Cells

## Contents
* README.md: This file, explaining the project
* [Final_project_presentation_v7.pdf](https://github.com/ravasconcelos/flowers_dcgan/blob/master/Final_project_presentation_v7.pdf): Project presentation in PDF format
* [Final_project_presentation_v7.pptx](https://github.com/ravasconcelos/flowers_dcgan/blob/master/Final_project_presentation_v7.pptx?raw=true): Project presentation in Power Point format
* [GeneratedFlowers.pdf](https://github.com/ravasconcelos/flowers_dcgan/blob/master/GeneratedFlowers.pdf): Some samples of generated flowers for each model 
* only_flowers.zip: Daisies from Flowers Recognition Kaggle Dataset
* dcgan_flowers_1.ipynb: Model inspired on Greg Surma' Drawing Cartoons with Generative Adversarial Networks, but with more layers
* dcgan_flowers_2.ipynb: Model inspired on Tensorflow DCGAN Tutorial
* dcgan_flowers_3.ipynb: Model 1, but with 1000 elements noise array
* dcgan_flowers_4.ipynb: Model 2, but with some hyperparemters from the Paper "Unsupervised Representation Learning With Deep Convolutional Generative Adversarial Networks" [4] 
* dcgan_flowers_4_BlackWhite.ipynb: Model 4, but the input images are in Black and White 
* dcgan_flowers_5.ipynb: Model inspired on the Paper "Unsupervised Representation Learning With Deep Convolutional Generative Adversarial Networks" [4] 
* dcgan_flowers_6.ipynb: Model 4, but with the first Convolutional layer using 2,2 stride. 

## Generated images

| ![](https://github.com/ravasconcelos/flowers_dcgan/blob/master/generated_images/model1/5000_epochs/download%20(3).png) | ![](https://github.com/ravasconcelos/flowers_dcgan/blob/master/generated_images/model2/2000_epochs/download%20(3).png) |  ![](https://github.com/ravasconcelos/flowers_dcgan/blob/master/generated_images/model3/2000_epochs/download.png) |
|:--:|:--:|:--:|
| *Model 1* | *Model 2* | *Model 3* |

| ![](https://github.com/ravasconcelos/flowers_dcgan/blob/master/generated_images/model4/1000_epochs/download%20(10).png) | ![](https://github.com/ravasconcelos/flowers_dcgan/blob/master/generated_images/model5/5000_epochs/index3.png) |  ![](https://github.com/ravasconcelos/flowers_dcgan/blob/master/generated_images/model6/6000_epochs/download.png) |
|:--:|:--:|:--:|
| *Model 4* | *Model 5* | *Model 6* |

More generated flowers separated by Model and epoch:
[GeneratedFlowers.pdf](https://github.com/ravasconcelos/flowers_dcgan/blob/master/GeneratedFlowers.pdf)

## Model Evolution
[![](https://img.youtube.com/vi/0uTfwXIWl40/0.jpg)](https://www.youtube.com/watch?v=0uTfwXIWl40)

## Generator/Discriminator Loss example
[![](https://github.com/ravasconcelos/flowers_dcgan/blob/master/generated_images/model6/6000_epochs/download%20(1).png)

## References
1. https://www.kaggle.com/alxmamaev/flowers-recognition
2. https://www.tensorflow.org/tutorials/generative/dcgan
3. https://machinelearningmastery.com/how-to-evaluate-generative-adversarial-networks/
4. https://arxiv.org/pdf/1511.06434.pdf
5. https://towardsdatascience.com/image-generator-drawing-cartoons-with-generative-adversarial-networks-45e814ca9b6b




