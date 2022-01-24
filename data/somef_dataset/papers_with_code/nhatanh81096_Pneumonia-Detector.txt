# Pneumonia Detection from Chest X-Rays

## Project Overview

In this project, you will apply the skills that you have acquired in this 2D medical imaging course  to analyze data from the NIH Chest X-ray Dataset and train a CNN to classify a given chest x-ray for the presence or absence of pneumonia. This project will culminate in a model that can predict the presence of pneumonia with human radiologist-level accuracy that can be prepared for submission to the FDA for 510(k) clearance as software as a medical device. As part of the submission preparation, you will formally describe your model, the data that it was trained on, and a validation plan that meets FDA criteria.

You will be provided with the medical images with clinical labels for each image that were extracted from their accompanying radiology reports. 

The project will include access to a GPU for fast training of deep learning architecture, as well as access to 112,000 chest x-rays with disease labels  acquired from 30,000 patients.

## Pneumonia and X-Rays in the Wild

Chest X-ray exams are one of the most frequent and cost-effective types of medical imaging examinations. Deriving clinical diagnoses from chest X-rays can be challenging, however, even by skilled radiologists. 

When it comes to pneumonia, chest X-rays are the best available method for diagnosis. More than 1 million adults are hospitalized with pneumonia and around 50,000 die from the disease every
year in the US alone. The high prevalence of pneumonia makes it a good candidate for the development of a deep learning application for two reasons: 1) Data availability in a high enough quantity for training deep learning models for image classification 2) Opportunity for clinical aid by providing higher accuracy image reads of a difficult-to-diagnose disease and/or reduce clinical burnout by performing automated reads of very common scans. 

The diagnosis of pneumonia from chest X-rays is difficult for several reasons: 
1. The appearance of pneumonia in a chest X-ray can be very vague depending on the stage of the infection
2. Pneumonia often overlaps with other diagnoses
3. Pneumonia can mimic benign abnormalities

For these reasons, common methods of diagnostic validation performed in the clinical setting are to obtain sputum cultures to test for the presence of bacteria or viral bodies that cause pneumonia, reading the patient's clinical history and taking their demographic profile into account, and comparing a current image to prior chest X-rays for the same patient if they are available. 

## About the Dataset

The dataset provided to you for this project was curated by the NIH specifically to address the problem of a lack of large x-ray datasets with ground truth labels to be used in the creation of disease detection algorithms. 

The data is mounted in the Udacity Jupyter GPU workspace provided to you, along with code to load the data. Alternatively, you can download the data from the [kaggle website](https://www.kaggle.com/nih-chest-xrays/data) and run it locally. You are STRONGLY recommended to complete the project using the Udacity workspace since the data is huge, and you will need GPU to accelerate the training process.

There are 112,120 X-ray images with disease labels from 30,805 unique patients in this dataset.  The disease labels were created using Natural Language Processing (NLP) to mine the associated radiological reports. The labels include 14 common thoracic pathologies: 
- Atelectasis 
- Consolidation
- Infiltration
- Pneumothorax
- Edema
- Emphysema
- Fibrosis
- Effusion
- Pneumonia
- Pleural thickening
- Cardiomegaly
- Nodule
- Mass
- Hernia 

The biggest limitation of this dataset is that image labels were NLP-extracted so there could be some erroneous labels but the NLP labeling accuracy is estimated to be >90%.

The original radiology reports are not publicly available but you can find more details on the labeling process [here.](https://arxiv.org/abs/1705.02315) 


### Dataset Contents: 

1. 112,120 frontal-view chest X-ray PNG images in 1024*1024 resolution (under images folder)
2. Meta data for all images (Data_Entry_2017.csv): Image Index, Finding Labels, Follow-up #,
Patient ID, Patient Age, Patient Gender, View Position, Original Image Size and Original Image
Pixel Spacing.


## FDA Submission

**Your Name: Nguyen Nhat Anh Vo**

**Name of your Device: Pneumonia Detector**

## Algorithm Description 

### 1. General Information

**Intended Use Statement:** Assisting radiologists in detecting pneumonia in Chest X-ray images with the view of PA/AP.

**Indications for Use:** It is well-used for both male and female from 1-100 years old. Patient can also exihit other diseases (Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Fibrosis, Hernia, Infiltration, Mass, Nodule, Pleural_Thickening, Pneumonia, Pneumothorax) in comorbid with pneumonia.

**Device Limitations:** Require high-power processing GPU card to run the algorithm.

**Clinical Impact of Performance:** 

### 2. Algorithm Design and Function

![process](./assets/Process.png)

**DICOM Checking Steps:** Perform 3 different DICOM checks
- Check patient position: Only AP or PA view will be processed
- Check image type (modality): Only DX type will be processed
- Check body part examined: Only chest taken image will be process

**Preprocessing Steps:** Rescale the image by dividing by 255.0, then normalize the image with the mean and standard deviation retrieved from the training data. Finally, resize the image to (1, 224, 224, 3) to fit in the network.

**CNN Architecture:**
- VGG19 model architecture is used for transfer learning. 

![vgg19](./assets/vgg19.png)

- Several custom layers are also added to the VGG19.

![my_model](./assets/my_model_plot.png)

### 3. Algorithm Training

**Parameters:**
- Augmentation used: 
    - Horizonal Flip
    - Height Shift Range = 0.1
    - Width Shift Range = 0.1
    - Rotation Range = 20
    - Shear Range = 0.1
    - Zoom Range = 0.1
* Batch size = 32
* Optimizer learning rate = 3e-4
* Layers of pre-existing architecture that were frozen: First 20 layers
* Layers of pre-existing architecture that were fine-tuned: None
* Layers added to pre-existing architecture:
    * Flatten
    * Dropout 0.5
    * Dense 1024, Activation = ReLU
    * Dropout 0.5
    * Dense 512, Activation = ReLU
    * Dropout 0.5
    * Dense 256, Activation = ReLU
    * Dense 1, Activation = Sigmoid

![train](./assets/ModelAcc.png)

![auc](./assets/AUC.png)

![pr](./assets/P-R.png)

**Final Threshold and Explanation:**

![f1-thresh](./assets/F1vsThres.png)

- Based on the F1-Score vs Threshold Chart, to balance between the Precision and Recall, the threshold of 0.728 will give the max value of F1-Score. 

### 4. Databases
- The databases contains 112,120 X-Ray images. The number of Pneumonia Positive images is only 1430 (1.27%).
- Therefore, to split the databases for training, I will have to:
  - Obtain all the postive cases of Pneumonia.
  - Divide the positive cases into 80%-20% for the Training and Validation Dataset.
 
**Description of Training Dataset:** 
- Balance the number of negative and positive cases in the training data.

**Description of Validation Dataset:** 
- Make the number of negative cases 4 times bigger the positive cases to somehow reflect the real-world clinical setting


### 5. Ground Truth

- The ground truth is NLP-derived labels. NLP at this stage is not complex enough to capture all the existing information of the reports. Hence, the accuracy is roughly 90%.

### 6. FDA Validation Plan

**Patient Population Description for FDA Validation Dataset:**
- Male and female patients in the age of 1 to 100. The gender distribution is slightly toward Male patient, the male to female ratio is approximately 1.2
- The patient may exihibit the following comorbid with Pneumonia: Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Fibrosis, Hernia, Infiltration, Mass, Nodule, Pleural_Thickening, Pneumonia, Pneumothorax - 
- The X-Ray Dicom file should has the following properties: 
    - Patient Postition: AP or PA
    - Image Type: DX
    - Body Part Examined: CHEST

**Ground Truth Acquisition Methodology:**
- Establish a silver standard of radiologist reading

**Algorithm Performance Standard:**

![f1-score](./assets/F1-Scores.png)

- The F1-Score should be approximately 0.435 to out-perform the current state-of-the-art method (CheXNet) [https://arxiv.org/pdf/1711.05225.pdf]

