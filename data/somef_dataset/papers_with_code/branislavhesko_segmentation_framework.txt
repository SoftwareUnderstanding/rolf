# Generic blood vessel segmentation in retinal fundus images

**This repository provides all the tools needed to perform image segmentation, the only thing needed should be the configuration setting located in config.py.**

One of the main problems in biomedical image processing is the lack of data. Especially for blood vessel segmentation,
only few public datasets are available. Moreover, this datasets are unhomogenous in size, which may pose problems for a range of approaches. In this repository you may find a simple yet generic approach to vessel segmentation. Images are processed by tiling blocks of a predefined size. There is no prerequisite for the image size.

## Models

Currently to support a model, you only need to add the model into **config.py** - *available_models* dictionary. Model should be constructed with a single parameter - number of classes. Actually we provide these models:

* U-Net - https://arxiv.org/pdf/1505.04597.pdf
* PSPNet - https://arxiv.org/pdf/1612.01105.pdf
* DeepLabv3p - https://arxiv.org/pdf/1802.02611.pdf
* CombineNet - our custom implementation. Model is larger, it is a combination of U-Net and PSPNet.

## Dataset

 To directly train you model using our segmentation framework, dataset should have this simple structure:

 ```
dataset_folder
|
|
|----train
        |
        |---imgs
        |---masks
|----eval
        |
        |---images
            |----img0
            |----imgX
        |---masks
            |----mask0
            |----maskX
 ```

Afterwards, you should fill these configuration entries in **config.py**:

* **FOLDER_WITH_IMAGE_DATA** - path to your dataset
* **NUM_CLASSES** - number of classes
* **CROP_SIZE**
* **OUTPUT** - path, where checkopoints are saved
* **OUTPUT_PATH** - experiment name

## Training

 pass
