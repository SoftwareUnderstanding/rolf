# SkinCheck - A deep-learning based Web-Application for Skin Cancer Detection

Skin Cancer is one of the most common forms of cancer, with increasing incidence. In Germany, it makes up one third of all cancer diagnoses (https://www.tk.de/techniker/gesundheit-und-medizin/praevention-und-frueherkennung/hautkrebs-fruehererkennung/hautkrebs-wie-hoch-ist-das-risiko-2015296). While Basal Cell Carcinoma (BCC) is the most frequent type of skin cancer, Melanoma is considered the most dangerous one. In both cases, however, early diagnosis is crucial to facilitate successfull treatment.

Since lesions occur on the surface of the skin, visual detection via dermatoscopy (an imaging technique that enables visualisation of deeper skin levels by removing surface reflection) is the best practice. With the widespread availability of high resolution cameras and even smartphone devices for skin monitoring (e.g. https://dermlite.com/products/dermlite-hud), research on automated analysis is growing. As an important step in the development of automated diagnostics, the International Skin Imaging Collaboration (ISIC) is hosting challenges on skin lesion analysis since 2016, providing the world's largest repository of standardized, publicly available dermatoscopic images.

This project aims to provide a web-application that classifies uploaded images of skin lesions into one of the following diagnostic labels:
1. Melanocytic Nevus (**nv**)
2. Dermatofibroma (**df**)
3. Melanoma (**mel**)
4. Basal Cell Carcinoma (**bcc**)
5. Actinic Keratosis (**aciec**)
6. Benign Keratosis (**bkl**)
7. Vascular lesions (**vasc**)

![Diagnostic Images](https://challenge2018.isic-archive.com/wp-content/uploads/2018/04/task3.png)


For this project, I used the HAM10000 ("Human against machine with 10000 images) dataset, a collection of 10015 dermatoscopic images that were made publibly available for the ISIC challenge, to train an artifical neural network to diagnose the pigmented skin lesions.
Since the risk of skin cancer increases with age and is higher in men, I included age and gender to improve accuracy.

**Please note:** Although I achieved a categorical accuracy of around 81%, the final model is heavily biased towards Nevi, lacks the appropriate sensitivity and specificity and thus should not (yet) be used for diagnostic purposes. Click [here](http://leonaha.pythonanywhere.com/) to try out my model!

## How did I approach this project?

### 1. Do some preprocessing

The distribution of image labels in the HAM10000 dataset are supposed to reflect the real-world setting. Thus, there's a strong categorical imbalance with around 6000 images showing benign melanocytic nevi and only 100-1000 images in the remaining categories.
To counteract class imbalance I did heavy image augmentation using **Keras ImageDataGenerator:**
<img width="733" alt="augmentation" src="https://user-images.githubusercontent.com/50407361/64116724-6206a500-cd93-11e9-8411-6a4e5ae14a7f.png">


### 2. Build the model(s)

With the web-application in mind, I decided to apply transfer learning using MobileNet, a convolutional neural network that uses depthwise separable convolutions and is thus more leightweight than other pretrained models (~ 4.5 million parameters; https://arxiv.org/pdf/1704.04861.pdf).

To feed the auxiliary input (age and gender) into the model and concatenate it with the extracted image features I had to write a custom image data generator (see Combined_model.ipynb).

For Hyperparameter-Finetuning, I applied **Cyclical Learning Rates** and the **One Cycle policy** (https://arxiv.org/pdf/1506.01186.pdf), a relatively new approach that improved my model's performance and that I'll definitely check out further. Instead of choosing a fixed or decreasing learning rate, you define a **minimum** and **maximum** LR. By allowing the learning rate to cyclically oscillate between the two values you avoid local minima and saddle points.

### 3. Create a web-application using Flask and TensorFlow.js

The (for now) final model was converted into TensorFlow.js format and integrated into a simple web-framework using Flask and Bootstrap. I created two additional pages containing visualisations of the data and the model's performance made with Plotly.

Click [here](http://leonaha.pythonanywhere.com/) to visit my web-app!

<img width="892" alt="skincheck" src="https://user-images.githubusercontent.com/50407361/64121956-92087500-cda0-11e9-8a36-e44186e1953f.png">

## Future Improvements

The unbalanced classes in the HAM10000 dataset are a challenging task that apparently wasn't solved by simply augmenting the underrepresented categories. There are, however, several strategies left that I might try out to address this difficulty, like:
- apply a loss function that punishes harder on false predictions in underrepresented classes
- downsampling overrepresented classes
- using model ensembles (this would however not be suited for web-applications or mobile devices)

## Original Data Source

### Original Challenge:
* https://challenge2018.isic-archive.com
* https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
[1] Noel Codella, Veronica Rotemberg, Philipp Tschandl, M. Emre Celebi, Stephen Dusza, David Gutman, Brian Helba, Aadi Kalloo, Konstantinos Liopyris, Michael Marchetti, Harald Kittler, Allan Halpern: “Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the International Skin Imaging Collaboration (ISIC)”, 2018; https://arxiv.org/abs/1902.03368
[2] Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Sci. Data 5, 180161 doi:10.1038/sdata.2018.161 (2018).
