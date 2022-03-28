# Image Translation using Pix2Pix GAN
In this repository,  I implemented the paper "Image-to-Image Translation with Conditional Adversarial Networks"(https://arxiv.org/pdf/1611.07004.pdf) to translate images from landscapes painting to Real-world images. 

## Table of contents
* [Folder Structure](#FolderStructure)
* [Results](#Results)
* [Acknowledgement](#Acknowledgement)
## Task Description

```
The task is to generate real images from landscape paintings.
```
Folder structure
--------------

```
├── datasets/Train/a       - this folder contains landscape images.
│   ├── image1001.png
│   └── image1002.png
│   └── --------------------
│
│
├── datasets/Train/b      - this folder contains Real-world images.
│   ├── image1001.png
│   └── image1002.png
│   └── --------------------  
│
├── datasets/Test-set/a             - this folder contains Test images(landscapes).
│   └── image1001.png
│   └── -------------------- 
│
├── save_model    -- this folder contains saved model
│
│── Python-scripts      - this folder contains  python files(can be run driectly in Jupyter notebook/IDE)
│
├──  train-MANET.py        - this file is used for training image.
│   
├──  testing.py         - this file is used for generating test images.
│   
├──  result        - this folder contains generated test images.
│ 
└──logs/tensorlogs     

```
## Results


<table>
  <tr>
    <td style="text-align: middle;">Input</td>
    <td style="text-align: middle;">Generated</td>
    <td style="text-align: middle;">Target</td>
  </tr>
  <tr>
    <td>
     <img src="https://raw.githubusercontent.com/Nisnab/Pix2Pix/main/dataset/facades/train/a/image1055.png" />
    </td>
    <td>
     <img src ="https://raw.githubusercontent.com/Nisnab/Pix2Pix/main/result/facades/image1055.png"/>
    </td>
    <td>
     <img src="https://raw.githubusercontent.com/Nisnab/Pix2Pix/main/dataset/facades/train/b/image1055.png"/>
    </td>
  </tr>
  <tr>
    <td>
     <img src="https://raw.githubusercontent.com/Nisnab/Pix2Pix/main/dataset/facades/train/a/image1056.png"/>
    </td>
    <td>
     <img src="https://raw.githubusercontent.com/Nisnab/Pix2Pix/main/result/facades/image1056.png"/>
    </td>
    <td>
     <img src="https://raw.githubusercontent.com/Nisnab/Pix2Pix/main/dataset/facades/train/b/image1056.png"/>
    </td>
  </tr>
  <tr>
    <td>
     <img src="https://raw.githubusercontent.com/Nisnab/Pix2Pix/main/dataset/facades/train/a/image1057.png"/>
    </td>
    <td>
     <img src="https://raw.githubusercontent.com/Nisnab/Pix2Pix/main/result/facades/image1057.png"/>
    </td>
    <td>
     <img src="https://raw.githubusercontent.com/Nisnab/Pix2Pix/main/dataset/facades/train/b/image1057.png"/>
    </td>
  </tr>
</table>

## Acknowledgement

Due to the fixed image size, the generated images are smaller in size. However, the generated images can be viewed in folders.

The repository borrows heavily fron (https://github.com/suhoy901/ImageTranslation)
