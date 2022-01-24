# :hospital: |Pneumonia| :hospital:
**Final Project** | Dani Eiroa | **IH BCN Data Analytics PT 2019**

## Overview
A pneumonia is an acute infection of the lung and is characterized by the appearance of fever and respiratory symptoms, together with the presence of lung opacities on Chest-X-Rays (CXR).

According to the Spanish Society or Radiology (SERAM), there are no reliable data on the number of (CXR) not reported in Spain, although there is a widespread conviction that, with exceptions, radiology services have never reported 100% of them. There are hospitals that, in fact, have stopped reporting CXR, as the workload has been inclined towards the reporting of more complicated techniques, such as CT and MRI.

Throughout the process described below, we aim to develop a tool that classifies a given CXR in one of this two classes: Normal and Pneumonia.

### Contents:
  - [Data Preparation](#data-preparation)
    - [Data Ingestion](#data-ingestion)
    - [Data Wrangling and Cleaning](#data-wrangling-and-cleaning)
  - [Data Analysis](#data-analysis)
    - [Data Exploration and Visualization](#data-exploration-and-visualization)
    - [Model Training and Evaluation](#model-training-and-evaluation)
  - [Conclusions](#conclusions)
  - [Problems - Setbacks](#problems---setbacks)
  - [Tools](#tools)
  - [References](#references)

## Data Preparation
The dataset is comprised of a total 26684 Chest X-Rays (CXR), some of them of patients affected by pneumonia and others with no pneumonia.

The dataset also contains two .csv files:
  - One with the patient Id (same as original image file name), a Target column (0 for normal and 1 for pneumonia) and the coordinates for the opacities consistent with pneumonia, as the original challenge included a segmentation task.
  - Another containing the class of the image (Opacity, Normal, Not-normal/No opacity) as well as patient Id.
  
The dataset was obtained from Kaggle and provided and labeled by the Radiological Society of North America (RSNA). [Link](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data)


#### General description of the dataset such as the size, complexity, data types, etc.
26684 [DICOM](https://es.wikipedia.org/wiki/DICOM) files. (3.53 GB)

     - stage_2_train_labels.csv (26684, 6) (1.49 MB)
        - Patient Id: nominal variable. File names correspond to patient Id.
        - X and Y: the center of the boxes for the segmentation part.
        - height and width: the dimension of the box for the segmentation part.
        - target: binary. 0-normal and 1-pneumonia.
        
    - stage_2_detailed_class_info.csv (26684, 2) (1.69 MB)
        - Patient Id: nominal variable. File names correspond to patient Id.
        - Class: nominal variable with three possible values:
            - 'Normal'
            - 'Not normal/Not pneumonia'
            - 'Pneumonia'

### **Data Ingestion:**
Dataset was downloaded and unzziped using python functions defined in all three files of [step 2](https://github.com/EiroaMD/final_project_IH_pneumonia/tree/master/2%20Image%20Management), as different strategies were carried out to try to solve the problem.

CSVs were loaded using pandas `.read_csv()` method and explored accordingly (see 'Data Wrangling and Cleaning').

### **Data Wrangling and Cleaning:**
The original kaggle challenge included a segmentation task. For this reason, there were columns showing the coordinates of the pneumonic opacities and patient Id's that were repeated (some X-rays contained more than one pneumonic opacities). Those columns and duplicate rows were dropped. The two `.csv` files were joined by patiend Id.

Using specific python libraries (see 'Tools' below), the following metadata from DICOM files were extracted and added to the dataframe: age, sex and X-ray [projection/view](https://en.wikipedia.org/wiki/Chest_radiograph#Views) (antero-posterior or postero-anterior).

The target column was already coded:
    - 0 - normal.
    - 1 - pneumonia.

The type column values were strings, so two strategies were carried out:
    - Numerical coding:
        - 0 - Normal
        - 1 - Not normal/Not pneumonia
        - 2 - Neumonia
    - Dummy creation:
        - Just in case the information had to be fed to the algorithm in that format.

As the dataset initial purpose was a challenge, the test subset is obviously not labeled. So the first thing that was carried out was creating three stratified subsets from the train folder, which was already labeled:
  - Training subset (80%)
  - Validation subset (10%)
  - Test subset (10%)

That process was done twice, stratifying both by [target (binary)](https://github.com/EiroaMD/final_project_IH_pneumonia/blob/master/2%20Image%20Management/2_file_classifier_t_v_t.ipynb) and [class (three classes)](https://github.com/EiroaMD/final_project_IH_pneumonia/blob/master/2%20Image%20Management/2_file_classifier_t_v_t_3_classes.ipynb). 

Also, after some setbacks described below, a third notebook doing the same process after [balancing](https://github.com/EiroaMD/final_project_IH_pneumonia/blob/master/2%20Image%20Management/2_file_classifier_t_v_t_balanced.ipynb) data was created.  Next, the images were copied into train/validation/test folders by creating as many folders as classes.

**In brief**, we ended this step with a clean dataframe shaped (26684, 12) and 26884 images divided 80-10-10% between train, validation, and testing, each folder containing either two (normal|pneumonia) or three (normal|not-normal/not-pneumonia|pneumonia) subfolders.

### **Data Storage:**
Throughout the process, a few `.csv`files were generated, to keep record of the files belonging to the different subsets. They are stored in the [`/data`](https://github.com/EiroaMD/final_project_IH_pneumonia/tree/master/data/csv) folder of the github repository.

The heft of the files was stored on a Google Cloud virtual machine with GPUs, where the  model training was carried out.
## Data Analysis
### **Data Exploration and Visualization**
An Exploratory Data Analysis was executed. Overall, data was clean and tidy. There were no NaNs and only five outliers in the only numerical variable present (age). After exploring the ages of the outliers (around 150 years old) and their CXRs, they were corrected by subtracting 100 years.

Analysis of the value counts of the different categorical variables was done. Special attention was given to the possible target variables, the binary (target) and the ternary (class). There are two possible classifications of the X-Rays: normal|pneumonia and normal|not-normal-not-pneumonia|pneumonia. The image below shows that any of the two possible classifications yields unbalanced data, which was a pain point during the whole process.

<p align="center">
  <img width="800" height="400" src="https://github.com/EiroaMD/final_project_IH_pneumonia/blob/master/6%20Images/target.png?raw=true">
</p>

Age was also plotted by grouping subjects depending on the target variable. 

<p align="center">
  <img width="500" height="250" src="https://github.com/EiroaMD/final_project_IH_pneumonia/blob/master/6%20Images/age_2.png?raw=true">
</p>

<p align="center">
  <img width="500" height="250" src="https://github.com/EiroaMD/final_project_IH_pneumonia/blob/master/6%20Images/age_3.png?raw=true">
</p>


Statistical analysis was carried out to check if the difference of mean age was significant:

  - For the binary target - **Welch's t-test**:

    - <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;H0:&space;\bar{x}&space;^{normal}&space;=&space;\bar{x}&space;^{pneumonia}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;H0:&space;\bar{x}&space;^{normal}&space;=&space;\bar{x}&space;^{pneumonia}" title="\large H0: \bar{x} ^{normal} = \bar{x} ^{pneumonia}" /></a>

    - <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;H1:&space;\bar{x}&space;^{normal}&space;\neq&space;\bar{x}&space;^{pneumonia}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;H1:&space;\bar{x}&space;^{normal}&space;\neq&space;\bar{x}&space;^{pneumonia}" title="\large H1: \bar{x} ^{normal} \neq \bar{x} ^{pneumonia}" /></a>

    - p-value of 3.27e-13. H0 is rejected. We can't conclude that there are no differences between mean age of people with pneumonia, compared to healthy subjects.

  - For the Ternary target - **ANOVA**:

    - <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;H0:&space;\bar{x}&space;^{normal}&space;=&space;\bar{x}&space;^{nnnp}&space;=&space;\bar{x}&space;^{pneumonia}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;H0:&space;\bar{x}&space;^{normal}&space;=&space;\bar{x}&space;^{nnnp}&space;=&space;\bar{x}&space;^{pneumonia}" title="\large H0: \bar{x} ^{normal} = \bar{x} ^{nnnp} = \bar{x} ^{pneumonia}" /></a>

    - <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;H1:&space;Means\hspace{2mm}&space;are\hspace{2mm}&space;not\hspace{2mm}&space;all\hspace{2mm}&space;equal." target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;H1:&space;Means\hspace{2mm}&space;are\hspace{2mm}&space;not\hspace{2mm}&space;all\hspace{2mm}&space;equal." title="\large H1: Means\hspace{2mm} are\hspace{2mm} not\hspace{2mm} all\hspace{2mm} equal." /></a>

    - p-value of 5.82e-90. H0 is rejected. We can't conclude that there are no differences between mean age of people with pneumonia, compared with the other two groups.

Initially, the three-category approach was decided, because the unbalance was not as big as with the binary approach, but a good result couldn't be achieved, the accuracy was not bad, but the rest of the metrics did not satisfy our standards and the model overfitted widely, so finally a binary classification with undersampling of the normal.

**In brief**, we ended up with a total of 12024 images (6012 normal - 6012 pneumonia) and a dataframe with a shape of (12024, 12), as well as three subsets of it (train, validation, and test).

### **Model Training and Evaluation**

After some research on the matter, we came to the conclusion that the best models for image classification tasks are [Convolutional Neural Networks (CNN)](https://towardsdatascience.com/convolutional-neural-network-17fb77e76c05)

Briefly, a CNN consists on an input, layers of filters that apply mathematical operations to the pixel values of the image (convolution and pooling) and  a cluster of fully connected or dense layers on the end of the network. This layers bound together produce an output, that in our cases is a category: normal or pneumonia.

Before feeding the images to the CNN, they had to be preprocessed using the `ImageDataGenerator` function from keras.

- Summary of the most important parameters used:
    - Activation functions:
        - ReLU.
        - Softmax for output layer.
    - Dropout:
        - A 20% Dropout was applied after the second convolution layer and after each fully conected layer.
    - Loss function: binary cross-entropy.
    - Optimizer: RMSProp
    - Learning rate: 0.001
    
A lot of parameter hypertuning was performed throughout the process, although the most relevant to the final results were the addition of dropout layers, which dramatically decreased overfitting and the selection of the [softmax](https://stats.stackexchange.com/a/410112) function on the output layer (the initial output function chosen was the sigmoid).

Although satisfying results are yielded by this model [(around 90% accuracy and F1-score)](https://github.com/EiroaMD/final_project_IH_pneumonia/blob/master/4%20Model%20Training/4_2_Model_Binary_from_directory_balanced.ipynb) with a low proportion of false negatives (which in this particular case would be worse than false positives - *i.e. it is better to treat a patient with no pneumonia that leave a sick patient with no treatment*), there was quite a bit of overfitting, despite extensive hypertuning.
 
For that reason it was decided to try to apply [Transfer Learning]([https://machinelearningmastery.com/transfer-learning-for-deep-learning/](https://machinelearningmastery.com/transfer-learning-for-deep-learning/)) techniques to the problem. Layers of the pretrained [VGG-19]([http://www.robots.ox.ac.uk/~vgg/](http://www.robots.ox.ac.uk/~vgg/)) network were used, combined with self-added fully connected layers.

- Summary of the most important parameters used:
    - Activation functions:
        - ReLU.
        - Softmax for output layer.
    - Dropout:
        - A 40% Dropout was applied after the first fully connected layer and an additional 20% after the second one.
    - Loss function: categorical cross-entropy.
    - Optimizer: SGD
    - Learning rate: 0.001

Same as we did before, parameter hypertuning was carried out, achieving the following [results](https://github.com/EiroaMD/final_project_IH_pneumonia/blob/master/4%20Model%20Training/4_4_Model_Binary_Transfer_VGG19_7_epochs.ipynb) after 7 epochs, with no overfitting:

<p align="center">
  <img width="600" height="400" src="https://github.com/EiroaMD/final_project_IH_pneumonia/blob/master/6%20Images/results.png?raw=true">
</p>
    
Afterwards, the model was tested both with images from the Test subset and with real-life X-rays downloaded from Google, as demonstrated in Steps [5_2](https://github.com/EiroaMD/final_project_IH_pneumonia/blob/master/5%20Testing/5_2_load_model_predict.ipynb) and [5_4](https://github.com/EiroaMD/final_project_IH_pneumonia/blob/master/5%20Testing/5_4_load_model_predict.ipynb).

## Conclusions
- We parted from a relatively clean dataset, and added some information extracted from DICOM tags.
- Although clean and tidy, the data presented a problem of imbalance, with a relatively low number of pneumonia images (6012/26684).
- There is a slight difference in terms of age, between subjects that have a normal X-Ray and subjects with pneumonia. That difference is statistically significant.
- After an iterative process, the imbalance was managed by undersampling.
- A CNN was built from scratch with relatively good performance (f1-score of 0.89), and a low number of false negatives, which in this particular case (a person with pneumonia leaving the hospital without treatment)
- Furthermore, with techniques of transfer-learning, another CNN was built and trained, achieving better metrics (f1-score of 0.92), and minimizing the overfitting.
## Problems - Setbacks
The main pain points throughout the process were:
- Dealing with unbalance: after trying to solve the problem with the unbalanced raw data and discarding the idea of oversampling, as we are dealing with medical images, I finally decided to do an undersampling of the normal images to balance both classes.
    - **Learning:** when dealing with classification problems, better quality than quantity. From now on, I'd rather have a small good, balanced dataset than a big one that I have to undersample anyway.
- DICOM: I learnt the hard way that `.dcm` files could not be fed to a CNN. I used some libraries (see "Tools") to convert and move images using Python.
    - **Learning:** DICOM is a very versatile format and you can do a lot of file management just by using Python.
- Accuracy is not everything. Actually, it's nothing: first time I trained the CNN, with unbalanced binary classes, I got an accuracy of almost 75%. It turned out that the CNN classified all X-rays as normal (75% of the train image set was normal).
    - **Learning:** looking at all the metrics is vital, as they provide different information and help to identify problems within the model.
- Overfitting: it never ceases to amaze me the difficulty of adequately dealing and correcting overfitting.
    - **Learning:** hypertune and iterate until you die.
## Tools:
**Environment**
- Google Cloud Platform Virtual Machine
    - 8 cores
    - 2 NVidia Tesla K80

**Libraries**
- **Numpy**
- **Pandas**
- **File management:** os, zipfile, shutil.
- **Image manipulation:** PyDicom, OpenCV, scikit-image, imutils.
- **Visualization:** Matplotlib, Seaborn.
- **Neural Networks:** TensorFlow|Keras.
- **Statistical Analysis:** SciPy Stats.
- **Metrics:** scikit-learn.
## References
[https://web.stanford.edu/class/cs231a/lectures/intro_cnn.pdf](https://web.stanford.edu/class/cs231a/lectures/intro_cnn.pdf)

[https://towardsdatascience.com/simply-deep-learning-an-effortless-introduction-45591a1c4abb](https://towardsdatascience.com/simply-deep-learning-an-effortless-introduction-45591a1c4abb)

[https://towardsdatascience.com/basics-of-image-classification-with-keras-43779a299c8b](https://towardsdatascience.com/basics-of-image-classification-with-keras-43779a299c8b)

[https://developers.google.com/machine-learning/practica/image-classification](https://developers.google.com/machine-learning/practica/image-classification?authuser=2)

[https://www.freecodecamp.org/news/how-to-build-the-best-image-classifier-3c72010b3d55/](https://www.freecodecamp.org/news/how-to-build-the-best-image-classifier-3c72010b3d55/)

[https://medium.com/merantix/deep-learning-from-natural-to-medical-images-74827bf51d6b](https://medium.com/merantix/deep-learning-from-natural-to-medical-images-74827bf51d6b)

[https://towardsdatascience.com/running-jupyter-notebook-in-google-cloud-platform-in-15-min-61e16da34d52](https://towardsdatascience.com/running-jupyter-notebook-in-google-cloud-platform-in-15-min-61e16da34d52)

[https://thispointer.com/python-how-to-move-files-and-directories/](https://thispointer.com/python-how-to-move-files-and-directories/)

[https://arxiv.org/abs/1409.1556](https://arxiv.org/abs/1409.1556)

[https://towardsdatascience.com/neural-style-transfer-4d7c8138e7f6](https://towardsdatascience.com/neural-style-transfer-4d7c8138e7f6)

[https://medium.com/@vijayabhaskar96/multi-label-image-classification-tutorial-with-keras-imagedatagenerator-cd541f8eaf24](https://medium.com/@vijayabhaskar96/multi-label-image-classification-tutorial-with-keras-imagedatagenerator-cd541f8eaf24)

[https://towardsdatascience.com/transfer-learning-in-tensorflow-9e4f7eae3bb4](https://towardsdatascience.com/transfer-learning-in-tensorflow-9e4f7eae3bb4)

[https://marubon-ds.blogspot.com/2017/09/vgg-fine-tuning-model.html](https://marubon-ds.blogspot.com/2017/09/vgg-fine-tuning-model.html)

[https://rdipietro.github.io/friendly-intro-to-cross-entropy-loss/](https://rdipietro.github.io/friendly-intro-to-cross-entropy-loss/)

[https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a](https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a)

[https://machinelearningmastery.com/how-to-calculate-precision-recall-f1-and-more-for-deep-learning-models/](https://machinelearningmastery.com/how-to-calculate-precision-recall-f1-and-more-for-deep-learning-models/)

[https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/](https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/)
