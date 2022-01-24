# EMsegmentation-mito
## Introduction

**U-Net** is a CNN used to segment areas of an image by class, and known for higher RAND index score of mitochodnria segmentation from electron microscopy images [1,2]. This repository provides the Python code for automatic segmentation of mitochondria using (2D) U-Net from EM images by [SNEMI3D](http://brainiac2.mit.edu/SNEMI3D/) [3].

The code for U-Net model was adapted from [this repositroy](https://github.com/YunYang1994/TensorFlow2.0-Examples) [4]. Keras has a built-in class `ImageDataGenerator` for data augmentation, but I have also tried to use the combination of more flexible tools: [`albumentations`](https://github.com/albumentations-team/albumentations) and [`ImageDataAugmentor`](https://github.com/mjkvaak/ImageDataAugmentor) (see [`seg_mito_albumentations.ipynb`](https://github.com/sumiya-kuroda/EMsegmentation-mito/blob/master/notebooks/seg_mito_albumentations.ipynb))[5,6]. 

<img src="https://github.com/sumiya-kuroda/EMsegmentation-mito/blob/master/misc/fig.png" alt="example" title="example">

## System Information

* OS Platform and Distribution: Linux Ubuntu 18.04
* GPU model and memory: 4 x Nvidia RTX 2080 Ti
* CUDA/cuDNN version: CUDA 10.1 / cuDNN 7.6.5
* Python version: 3.6.9
* TensorFlow version: 2.1.0
* TensorFlow installed from: pipenv install tensorflow==2.1.0

## Usage

Install packages on `requirements.txt` or `Pipfile` from the command line, with whatever package managers you prefer. The instruction below was using `pipenv`.

```sh
$ pip install pipenv
$ pipenv install
# use pipenv graph to check installed packages
```

### python
Once you installed packages, you can start training and testing with following command. They use `dataset/SNEMI3D_mito` data by default.

```sh
# training
$ pipenv run python3 training_mito.py
# test
$ pipenv run python3 test_mito.py
```

At the same time, you can try segmentaion of your own files (after saving your U-Net model to a HDF file). Put your EM images into the `dataset` direcotry, and use `prediction_mito.py`.

```sh
$ pipenv run python3 prediction_mito.py --input dataset/demo
```

### Jupyter Notebook
Alternatevily, you can train and test with Jupyter Notebook. `seg_mito_keras.ipynb` simply uses `ImageDataGenerator` to augment images, whereas `seg_mito_albumentations.ipynb` takes advantage of `albumentations` and `ImageDataAugmentor` for advanced operations. However, there seems to be a version conflict between `TensorFlow` and `ImageDataAugmentor`. 

## Dataset Description

`SNEMI3D_mito` contains 50 training images, 20 validation iamges, and 30 test images. Each of them is an 8-bit grayscale png file and has a size of 1024 x 1024 pixels. [CLAHE](https://imagej.net/Enhance_Local_Contrast_(CLAHE)) was applied beforehand to prevent the amplification of noise. Below is the directory structure of `SNEMI3D_mito`, and when you input your own EM images to the U-Net, your datasets should have the same structure.

```
SNEMI3D_mito/
├── train/
│   ├── images/ 
│   │    ├── 0000.png
│   │    ├── 0001.png
│   │    ├── ...
│   │    └── 0049.png
│   └── labels/
│        ├── 0000.png
│        ├── 0001.png
│        ├── ...
│        └── 0049.png
├── valid/
│   ├── images/ 
│   │    ├── 0050.png
│   │    ├── 0051.png
│   │    ├── ...
│   │    └── 0069.png
│   └── labels/ 
│        ├── 0050.png
│        ├── 0051.png
│        ├── ...
│        └── 0069.png
└── test/
    ├── images/ 
    │    ├── 0070.png
    │    ├── 0071.png
    │    ├── ...
    │    └── 0099.png
    └── labels/
         ├── 0070.png
         ├── 0071.png
         ├── ...
         └── 0099.png
```

## References
1. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. International Conference on Medical image computing and computer-assisted intervention, 234-241. ([link](https://arxiv.org/abs/1505.04597))

```
@misc{ronneberger2015unet,
    title={U-Net: Convolutional Networks for Biomedical Image Segmentation},
    author={Olaf Ronneberger and Philipp Fischer and Thomas Brox},
    year={2015},
    eprint={1505.04597},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

2. Urakubo, H., Bullmann, T., Kubota, Y., Oba, S., & Ishii, S., (2019). UNI-EM: An Environment for Deep Neural Network-Based Automated Segmentation of Neuronal Electron Microscopic Images. Scientific Reports 9, 19413. ([link](https://www.nature.com/articles/s41598-019-55431-0))

```
@article{urakubo_bullmann_kubota_oba_ishii_2019,
  title={UNI-EM: An Environment for Deep Neural Network-Based Automated Segmentation of Neuronal Electron Microscopic Images},
  author={Urakubo, Hidetoshi and Bullmann, Torsten and Kubota, Yoshiyuki and Oba, Shigeyuki and Ishii, Shin},
  year={2019},
  DOI={10.1101/607366}
}
```

3. Arganda-Carreras, I., Seung, S., H., Vishwanathan, A., & Berger, D., R. (2013). SNEMI3D: 3D Segmentation of neurites in EM images. ISBI 2013. ([link](http://brainiac2.mit.edu/SNEMI3D/home))

```
@article{kasthuri2015saturated,
  title={Saturated reconstruction of a volume of neocortex},
  author={Kasthuri, Narayanan and Hayworth, Kenneth Jeffrey and Berger, Daniel Raimund and Schalek, Richard Lee and Conchello, Jos{\'e} Angel and Knowles-Barley, Seymour and Lee, Dongil and V{\'a}zquez-Reina, Amelio and Kaynig, Verena and Jones, Thouis Raymond and others},
  journal={Cell},
  volume={162},
  number={3},
  pages={648--661},
  year={2015},
  publisher={Elsevier}
}
```

4. YunYang1994. (2019). U-Net: Convolutional Networks for Biomedical Image Segmentation. GitHub repository. https://github.com/YunYang1994/TensorFlow2.0-Examples
```
@Github_Project{TensorFlow2.0-Examples,
  title={U-Net: Convolutional Networks for Biomedical Image Segmentation},
  author={YunYang1994},
  email={www.dreameryangyun@sjtu.edu.cn},
  url={https://github.com/YunYang1994/TensorFlow2.0-Examples},
  year={2019},
}
```

5. albumentations-team. (2020). albumentations. GitHub repository. https://github.com/albumentations-team/albumentations
```
@Article{info11020125,
    author = {Buslaev, Alexander and Iglovikov, Vladimir I. and Khvedchenya, Eugene and Parinov, Alex and Druzhinin, Mikhail and Kalinin, Alexandr A.},
    title = {Albumentations: Fast and Flexible Image Augmentations},
    journal = {Information},
    volume = {11},
    year = {2020},
    number = {2},
    article-number = {125},
    URL = {https://www.mdpi.com/2078-2489/11/2/125},
    ISSN = {2078-2489},
    DOI = {10.3390/info11020125}
}
```

6. Tukiainen, M. (2019). ImageDataAugmentor. GitHub repository. https://github.com/mjkvaak/ImageDataAugmentor

```
@misc{mjkvaak_aug,
  author = {Tukiainen, M.},
  title = {ImageDataAugmentor},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {https://github.com/mjkvaak/ImageDataAugmentor/}
}
```
## Acknowledgements

My thanks are due to members in Yoshiyuki Kubota lab: in particular to Hidetoshi Urakubo for helpful discussion and preparing mitochondria labels from the SNEMI dataset. 

