# Human Face Generation
This repo is me playing around with basic GANS introduced in thi paper https://arxiv.org/pdf/1406.2661.pdf.

## Table of Contents
1. [Imports](#Imports) 
2. [What you need](#What-you-need)
3. [Greyscale Results](#Greyscale-Results)
4. [Colored Results](#Colored-Results)
5. [Additional Notes](#Additional-Notes)

## Imports
```
tensorflow                2.2.0
opencv-python             4.3.0.36
numpy                     1.18.5
matplotlib                3.2.1
```

## What you need
- A dataset with the following dimentions with dimentions bigger than (144, 144), the dataset I used had the following dimentions HEIGHT, WIDTH = (218, 178). Then the program cropped the image to be the needed size.
    - I used the following https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8
    - The dataset goes in ./humans/img_align_celeba
    - Note that the data set is pretty big (5 million celeb images), I only used around 3000 to train the model

## Grayscale Results
<img src="./assets/grey_tile.jpg" alt="drawing" width=800px>

## Colored Results
<img src="./assets/color_tile.jpg" alt="drawing" width=800px>

## Additional Notes
- The program ran for around 500 epochs
- Better results could be achieved with better stability, I used the following guide to improve the stability https://github.com/soumith/ganhacks
- Additionally, a better varient of the GAN could be used to reach better results such as WGAN
