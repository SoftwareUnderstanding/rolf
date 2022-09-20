
<img src=https://see.fontimg.com/api/renderfont4/KpAp/eyJyIjoiZnMiLCJoIjoxMDAsInciOjEwMDAsImZzIjoxMDAsImZnYyI6IiMwRjlCRkEiLCJiZ2MiOiIjMEMwMDAwIiwidCI6MX0/QXV0b1NvdXJjZUlELUxpZ2h0/kg-second-chances-sketch.png>


[![DOI](https://zenodo.org/badge/440851447.svg)](https://zenodo.org/badge/latestdoi/440851447) 
<a href="https://ascl.net/2203.014"><img src="https://img.shields.io/badge/ascl-2203.014-blue.svg?colorB=262255" alt="ascl:2203.014" /></a>

<img src="https://github.com/FiorenSt/AutoSourceID-Light/blob/main/Plots/OpticalImagePatch.png " width=50% height=50%><img src="https://github.com/FiorenSt/AutoSourceID-Light/blob/main/Plots/LoGOnOptical.png " width=50% height=50%> 

<!--
![GitHub stars](https://img.shields.io/github/stars/FiorenSt/AutoSourceID-Light?style=social)
![GitHub forks](https://img.shields.io/github/forks/FiorenSt/AutoSourceID-Light?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/FiorenSt/AutoSourceID-Light?style=social)
![GitHub followers](https://img.shields.io/github/followers/FiorenSt?style=social)
-->



# Description
With the ever-increasing survey speed of optical wide-field telescopes, rapid and reliable source localization is paramount. We propose a new way to analyze optical imaging
data that uses computer vision techniques that can naturally deal with large amounts of data. We present AutoSourceID-Light (ASID-L), an innovative framework for rapidly localizing sources in optical images.

## Table of Contents 
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Features](#features)
- [Credits](#credits)


# Installation


_Follow the instructions below to download and start using ASID-L._

1. Clone the repo
   ```sh
   git clone https://github.com/FiorenSt/AutoSourceID-Light.git
   ```
2. Download the Zenodo folder for training/test/validation sets    
   [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5902893.svg)](https://doi.org/10.5281/zenodo.5902893)

3. Save the files in a folder "TrainingSet" and include the folder in the ASID-L repository
4. Create an empty folder "RESULTS" 

# Dependencies:
<!--
<img src="https://img.shields.io/badge/python%20-%2314354C.svg?&style=for-the-badge&logo=python&logoColor=white&style=plastic"/> ![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white&style=plastic)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white&style=plastic)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white&style=plastic)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white&style=plastic)
-->

* Python 3 (or superior)
* TensorFlow 2 
* Scikit-Image 0.18.1
* Numpy 1.20.3
* Joblib 1.0.1
* Patchify 0.2.3
* OpenCV 4.5.1
* Astropy 4.2.1

This combination of package versions works on most Linux and Windows computers, however other package versions may also work.
If the problem persist, raise an issue and we will help you solve it.


# Usage

The use of the pre-trained ASID-L is straight forward: 

```
python ASID-L.py
```

It loads a .fits image and the pre-trained model, and it outputs a catalog 'coordinates.txt' in the folder 'RESULTS'.

**Other parameters:**
 
-DATA_PATH './TrainingSet/ML1_20200601_191800_red_cosmics_nobkgsub.fits'  **_(path of the .fits image)_**

-MODEL_PATH './MODELS/TrainedModel.h5'   **_(path of the model)_**

-demo_plot   **_(shows a plot with an optical patch superimposed with the locations of the sources in red)_**

-CPUs  **_(number of CPUs for parallel processing)_**

Here an example,
```
python ASID-L.py -DATA_PATH './TrainingSet/ML1_20200601_191800_red_cosmics_nobkgsub.fits' -MODEL_PATH './MODELS/TrainedModel.h5' -demo_plot
```

### Train U-Net from scratch

 To train the U-Net without additional changes run:
 ```
 python ASID-L.py -train_model
 ```
 You will find the trained model in the folder '/MODELS/FROM_SCRATCH'. You can then run the pre-trained version of ASID-L with -MODEL_PATH your new trained model.
 
**Other parameters:**

-snr_threshold **_(SNR cut-off for the training set)_** 

-epochs **_(the number of epochs)_**



 
# License
Copyright 2022 Fiorenzo Stoppa

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.




# Features

An open question that we want to address in the future is how the resolution of the images affects the localization results.
A first promising test can be found below, where we applied ASID-L, trained on MeerLICHT images, to images from the Hubble Space Telescope. The latter has a Full-Width at Half-
Maximum (FWHM) PSF of about 0.11 arcseconds, much better than the 2-3 arcseconds of MeerLICHT.

<img src="https://github.com/FiorenSt/AutoSourceID-Light/blob/main/Plots/HSTField10396.png " >

Star cluster image retrieved from the Hubble Space Telescope archive (GO-10396, PI: J.S. Gallagher). The red circles in the zoomed windows are the locations of the sources identified by ASID-L.

Although this is an early study, it appears that ASID-L is capable of localizing  sources without the need to re-train the U-Net on HST images. The main difference between MeerLICHT and HST, the resolution of the images, does not seem to affect the results of the method. 


# Credits
Credit goes to all the authors of the paper: 

**_AutoSourceID-Light. Fast Optical Source Localization via U-Net and Laplacian of Gaussian_**
