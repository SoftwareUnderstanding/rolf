# Pneumonia Detection From X-Rays
This repository contains a completed cap-stone project for Udacity's "Applying AI to 2D Medical Imaging Data" course, 
part of the AI for Healthcare Nanodegree program.  It has been reviewed by Udacity instructors and met project specifications.

**Introduction**  
Advancements in deep learning and computer vision allow new opportunities to create software to assist medical
physicians.  Assistive software can improve patient prioritization or reduce physicians' efforts to examine medical images.
In this project, computer vision with a convolutional neural network (CNN) model is trained to predict the presence 
or absence of pneumonia from chest X-Ray images. The VGG16 CNN model was fine-tuned for this classification task. The intended use for this model is to pre-screen chest X-Ray images prior to radiologists' review to reduce their workload.  

The paper of Pranav Rajpurkar et al. (2017), "CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning", 
provides benchmarks to compare pneumonia classification performance against.  This paper established F1-scores as the metric to compare radiologists' and algorithms' 
performance in identifying pneumonia(Wang et al., 2017). 
F1-scores are the harmonic average of the precision and recall of a model's predictions against ground truth labels.
In a subset of 420 images from the ChestX-ray14 dataset, the CheXNet algorithm achieved an F1 score of 0.435, while a panel of four independent Radiologists averaged an F1 score of 0.387. 
This project's final F1 score is 0.36, which is similar in performance to the panel of radiologists. 

This project is organized in three Jupyter Notebooks:  
- 1_EDA (Exploratory Data Analysis): NIH X-Ray Dataset metadata analysis and X-ray image pixel-level analysis. 
- 2_Build_and_Train_Model: Image pre-processing with Keras ImageDataGenerator, split dataset using Scikit-Learn, build & train a Keras Sequential model, 
and convert probabilistic outputs to binary predictions.  
- 3_Inference:  DICOM pixel data extraction, normalize & standardize pixel data, and apply trained model to make predictions.

![test1.dcm](out/Example_test1.JPG)  
**Figure 1.** Example of in-line prediction output in `3_Inference.ipynb` Jupyter Notebook 

**References**  
[1]  Pranav Rajpurkar, Jeremy Irvin, Kaylie Zhu, Brandon Yang, Hershel Mehta, Tony Duan, Daisy Ding, Aarti Bagul, Curtis Langlotz, Katie Shpanskaya, Matthew P. Lungren, Andrew Y. Ng, "CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning,"  arXiv:1711.05225, Dec 2017. [Link](https://arxiv.org/abs/1711.05225)   
[2]  Xiaosong Wang, Yifan Peng, Le Lu, Zhiyong Lu, MohammadhadiBagheri, Ronald M. Summers. "ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases", IEEE CVPR, pp. 3462-3471,2017 [Link](https://openaccess.thecvf.com/content_cvpr_2017/papers/Wang_ChestX-ray8_Hospital-Scale_Chest_CVPR_2017_paper.pdf)


### Dataset
This project uses the ChestX-ray14 dataset curated by Wang et al. and released by NIH Clinical Center. 
It is comprised of 112,120 X-Ray images with disease labels from 30,805 unique patients.  
The disease labels for each image were created using Natural Language Processing (NLP) to process 
associated radiological reports for fourteen common pathologies. The estimated accuracy of the NLP labeling accuracy is estimated to be >90%.

**References**  
[1]  Xiaosong Wang, Yifan Peng, Le Lu, Zhiyong Lu, MohammadhadiBagheri, Ronald M. Summers. "ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases", IEEE CVPR, pp. 3462-3471,2017 [Link](https://openaccess.thecvf.com/content_cvpr_2017/papers/Wang_ChestX-ray8_Hospital-Scale_Chest_CVPR_2017_paper.pdf)

## Getting Started

1. Set up your Anaconda environment.  
2. Clone `https://github.com/ElliotY-ML/Pneumonia_Detection_ChestX.git` GitHub repo to your local machine.
3. Open `1_EDA.ipynb` with Jupyter Notebook for exploratory data analysis.
4. Open `2_Build_and_Train_Model.ipynb` with Jupyter Notebook for image pre-processing with Keras ImageDataGenerator, 
ImageNet VGG16 CNN model fine-tuning, and threshold analysis.
5. Open `3_Inference.ipynb` with Jupyter Notebook for inference with a DICOM file.
6. Complete project results discussion can be found in `FDA_Preparation.md`.

### Dependencies  
Using Anaconda consists of the following:

1. Install [`miniconda`](http://conda.pydata.org/miniconda.html) on your computer, by selecting the latest Python version for your operating system. If you already have `conda` or `miniconda` installed, you should be able to skip this step and move on to step 2.
2. Create and activate * a new `conda` [environment](http://conda.pydata.org/docs/using/envs.html).

\* Each time you wish to work on any exercises, activate your `conda` environment!

---

## 1. Installation

**Download** the latest version of `miniconda` that matches your system.

|        | Linux | Mac | Windows | 
|--------|-------|-----|---------|
| 64-bit | [64-bit (bash installer)][lin64] | [64-bit (bash installer)][mac64] | [64-bit (exe installer)][win64]
| 32-bit | [32-bit (bash installer)][lin32] |  | [32-bit (exe installer)][win32]

[win64]: https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86_64.exe
[win32]: https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86.exe
[mac64]: https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
[lin64]: https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
[lin32]: https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86.sh

**Install** [miniconda](https://docs.conda.io/en/latest/miniconda.html) on your machine. Detailed instructions:

- **Linux:** https://docs.conda.io/en/latest/miniconda.html#linux-installers
- **Mac:** https://docs.conda.io/en/latest/miniconda.html#macosx-installers
- **Windows:** https://docs.conda.io/en/latest/miniconda.html#windows-installers

## 2. Create and Activate the Environment

For Windows users, these following commands need to be executed from the **Anaconda prompt** as opposed to a Windows terminal window. For Mac, a normal terminal window will work. 

#### Git and version control
These instructions also assume you have `git` installed for working with GitHub from a terminal window, but if you do not, you can download that first with the command:
```
conda install git
```

**Create local environment**

1. Clone the repository, and navigate to the downloaded folder. This may take a minute or two to clone due to the included image data.

```
git clone https://github.com/ElliotY-ML/Pneumonia_Detection_ChestX.git
cd Pneumonia_Detection_ChestX
```

2. Create and activate a new environment, named `ChestX-Pneumonia` with Python 3.8.  Be sure to run the command from the project root directory since the environment.yml file is there.  If prompted to proceed with the install `(Proceed [y]/n)` type y and press `ENTER`.

	- __Linux__ or __Mac__: 
	```
	conda env create -f environment.yml
	source activate ChestX-Pneumonia
	```
	- __Windows__: 
	```
	conda env create -f environment.yml
	conda activate ChestX-Pneumonia
	```
	
	At this point your command line should look something like: `(ChestX-Pneumonia) <User>:USER_DIR <user>$`. The `(ChestX-Pneumonia)` indicates that your environment has been activated.


## Repository Instructions

Udacity's original project instructions can be read in the `Project_Overview.md` file.

**Project Overview**

   1. Exploratory Data Analysis
   2. Building and Training Your Model
   3. Inference
   4. FDA Preparation

### Part 1: Exploratory Data Analysis  

Open `1_EDA.ipynb` with Jupyter Notebook for exploratory data analysis.  The following data are examined:
1.  ChestX-ray14 Dataset metadata contains information for each X-Ray image file, the associated disease findings, patient gender, age, patient position during X-ray, and image shape.
2.  Pixel level assessment of X-Ray image files by graphing Intensity Profiles of normalized image pixels.  X-Rays are also displayed using scikit-image.


### Part 2: Building and Training Your Model, Fine Tuning Convolutional Neural Network VGG16 for Pneumonia Detection from X-Rays  

Inputs:
- ChestX-ray14 dataset containing 112,120 X-Ray images (.png) in `data/images` and metadata in `data/Data_Entry_2017.csv` file [1].  
**NOTE:** The dataset is not included in this GitHub repo, because the dataset size is greater than 42GB.  Please download a copy of the dataset from [https://nihcc.app.box.com/v/ChestXray-NIHCC](https://nihcc.app.box.com/v/ChestXray-NIHCC) and unpack into `/data/images`.

Output:
- CNN model trained to classify a chest X-Ray image for presence or absence of pneumonia in `/out/my_model1.json`. 
- `/out/xray_class_my_model.best.hdf5` containing model weights.  
**NOTE:** This is not included in this GitHub repo.

1.  Open `2_Build_and_Train_Model` with Jupyter Notebook.
2.  Create training data and validation data splits with scikit-learn train_test_split function.
3.  Ensure training data split is balanced for positive and negative cases.  Ensure validation data split has a positive to negative case ratio that reflects clinical scenarios.  Also check that each split has demographics that are reflective of the overall dataset.
4.  Prepare image preprocessing for each data split using Keras ImageDataGenerator.
5.  To fine-tune the ImageNet VGG16 model, create a new Keras Sequential model by adding VGG16 model layers and freezing their ImageNet-trained weights.  Subsequently add Dense and Dropout layers, which will have their weights trained 
for classifying chest X-Ray images for pneumonia.
6.  The model training will have a history to show loss metrics at each training epoch.  The best model weights are also captured at each training epoch.
7.  Model predictions initially return as probabilities between 0 and 1.  These probabilistic results were compared against ground truth labels.  
8.  A threshold analysis was completed to select the boundary at which probabilistic results are converted into binary results of either pneumonia presence or absence.
 
The CheXNet algorithm achieved an F1 score of 0.435, while a panel of four independent Radiologists averaged an F1 score of 0.387 [2]. 
This project's final F1 score is 0.36, which is similar in performance to the panel of Radiologist. 

**References**  
[1]  Xiaosong Wang, Yifan Peng, Le Lu, Zhiyong Lu, Mohammadhadi Bagheri, Ronald M. Summers. "ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases", IEEE CVPR, pp. 3462-3471,2017 [Link](https://openaccess.thecvf.com/content_cvpr_2017/papers/Wang_ChestX-ray8_Hospital-Scale_Chest_CVPR_2017_paper.pdf)  
[2]  Pranav Rajpurkar, Jeremy Irvin, Kaylie Zhu, Brandon Yang, Hershel Mehta, Tony Duan, Daisy Ding, Aarti Bagul, Curtis Langlotz, Katie Shpanskaya, Matthew P. Lungren, Andrew Y. Ng, "CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning,"  arXiv:1711.05225, Dec 2017. [Link](https://arxiv.org/abs/1711.05225)   


### Part 3: Inference  
The [`3_Inference Jupyter Notebook`](https://github.com/ElliotY-ML/Pneumonia_Detection_ChestX/blob/master/3_Inference.ipynb)
contains the functions to load DICOM files, pre-process DICOM image, load the model built in 2_Build_and_Train_Model, and predict the presence of pneumonia from the DICOM image.

Inputs:
- .dcm DICOM medical imaging file, contains metadata and a medical image

Output:
- DICOM image is displayed with a prediction of whether the patient is Positive or Negative for Pneumonia

The following steps should be performed to analyze a chest X-Ray DICOM file:
1.  Load DICOM file with `check_dicom(filename)` function.  It's output is the DICOM pixel_array or an error message if the DICOM file is not a Chest X-Ray.    
2.  Pre-process the loaded DICOM image with `preprocess_image(img=pixel_array, img_mean=0, img_std=1, img_size=(1,224,224,3))` function.
3.  Load trained model with `load_model(model_path, weight_path)`.
4.  Make prediction with `predict_image(model, img, thresh=0.245)`.




### Part 4: FDA Preparation  
Complete project results discussion can be found in `FDA_Preparation.md`

## License

This project is licensed under the MIT License - see the [LICENSE.md](./LICENSE.md)
