# Kaggle-APTOS-2019-Blindness
Top 37% entry (out of 2943) for Kaggle APTOS 2019 Blindness competition

The goal of this competition was to identify and rate the level of diabetic retinopathy from retinal images.  My best approach was an ensemble of deep learning classifiers (resnet34, efficientNetB6) trained on both original and preprocessed images.  Model ratings scored a kappa metric of 0.904 on a test set of approx. 12,000 images.  

More info can be found at the Kaggle site: https://www.kaggle.com/c/aptos2019-blindness-detection/overview

## Solution

### Data
Retinopathy causes changes in a healthy retina that can be seen in close-up retinal images.  Clinicians can assess the level of illness from characteristics of the image.

![Data](https://github.com/filipmu/Kaggle-APTOS-2019-Blindness/blob/master/doc_images/diabetic%20retinopathy%201.png)

Source: https://www.biorxiv.org/content/biorxiv/early/2018/06/19/225508.full.pdf

A clinician rates the presence of diabetic  retinopathy in each image on a scale of 0 to 4, according to International Clinical Diabetic Retinopathy severity scale (ICDR):
* 0 – No DR
* 1 – Mild DR
* 2 – Moderate DR
* 3 – Severe DR
* 4 – Proliferative DR
Ratings are based on human judgement of images with differing levels of brightness, orientation, and focus so there is some variation in the ratings.

Kaggle provided a data set of 3660 training images with ratings.  In addition I found a number of other data sets available online to add to the training set.  In all, a total of 44,000 image samples were obtained.  
### Evaluation Metric
Submissions are scored based on the quadratic weighted kappa, which measures the agreement between two ratings. This metric typically varies from 0 (random agreement between raters) to 1 (complete agreement between raters). In the event that there is less agreement between the raters than expected by chance, this metric may go below 0. The quadratic weighted kappa is calculated between the scores assigned by the human rater and the predicted scores.

### Data Selection
The proportion of samples with a 0 - No DR rating was significantly higher than the other ratings.  Early results showed better prediction on a validation set if the images with 0 ratings were excluded from the training set.  In order to accomplish this the Kaggle supplied 0 rated images were retained, and the other 0 rated images were excluded from further training.  In addition it was found that a number of images were duplicated and so duplicates were removed as well.  This resulted in a training set of roughly 12,000 images.

### Image Preprocessing
In order to compensage for brightness changes and to increase the contrast of the various indicators of retinopathy in the image a few preprocessing techniques were used, leveraging the opencv python library cv2.

#### Increased contrast
The first was taken from this starter example: https://www.kaggle.com/ratthachat/aptos-eye-preprocessing-in-diabetic-retinopathy.  In this example, contrast is increased by subtracting a blurred image from the original.

![Increased Contrast](https://github.com/filipmu/Kaggle-APTOS-2019-Blindness/blob/master/doc_images/blur%20contrast%20images.png)

Additional image processing was also used in the models:

#### Contrast Limited Adaptive Histogram Equalization (CLAHE) method -on RGB
Contrast increased using CLAHE independently on each channel of the RGB image.

python code from https://github.com/keepgallop/RIP/blob/master/RIP.py

![CLAHE](https://github.com/filipmu/Kaggle-APTOS-2019-Blindness/blob/master/doc_images/clahe%20processed.png)

#### CLAHE using CIELAB colorspace
In this approach the RGB image is converted to the L*A*B* color space and CLAHE is used to increase contrast on only the Lightness (L) channel.  This is motivated by the fact that CIELAB was designed so that a numerical value change corresponds to amount of perceived change. Using this approach preserves global image color while normalizing contrast on L.

![CLAHEL](https://github.com/filipmu/Kaggle-APTOS-2019-Blindness/blob/master/doc_images/clahel%20processed.png)

### Data Augmentation

The training images were transformed at random during training to augment the data.  Images were flipped left to right, rotated from 0 - 45 degrees, size adjusted from 100%-110%, brightness varied 100-110%.

### Convolutional Neural Networks
Two differerent architecture families were leveraged in this effort:

#### Resnet
The traditional residual NN that is pretrained on imagenet data.  https://pytorch.org/hub/pytorch_vision_resnet/
Based on preliminary training and validation, Resnet18 was satisfactory, Resnet34 provided significant improvement, and Resnet50 did not provide substantial increase in validation accuracy.  Resnet34 was used going forward.

#### Efficientnet
An improved residual NN architecture that has better scaling properties (using less compute resources for similar imagenet performance as resnet) Paper: https://arxiv.org/abs/1905.11946  Pytorch implementation: https://github.com/lukemelas/EfficientNet-PyTorch

EfficientNet-B6 was used based on success of other competitors.

#### Approach for turning an Ordinal Regression problem into a Classification problem 
Predicting Retinopathy ratings is an ordinal regression problem, since valid ratings are integers 0-4 and have an inherent order.  Convolutional neural network architectures are designed as classifiers.  if the ratings are used as classes predictions will not take advantage of the inherent order in the classes.  The model predicted probabilities for the classes might not make sense for ordinal data.  For example if rating 0 and rating 4 both have high probability of being correct while 1,2, or 3 are lower.

The apriori information that the ratings are ordinal can be encoded in a new definition of output classes of the training data.  In this case, it becomes a multi-label problem.
https://www.cs.waikato.ac.nz/~eibe/pubs/ordinal_tech_report.pdf

Mapping from rating to new class Labels:

|Condition on Rating|New Class|
|----|---|
|r=0|   |
|r>=1| 1 |
|r>=2| 2|
|r>=3|3|
|r>=4|4|

Examples of applying the mapping rule to generate mult-class labels

|Data sample Rating|Data sample New Class Labels|
|----|---|
|0| ''|
|1 |'1'|
|2 |'1,2'|
|3 |'1,2,3'|
|4 |'1,2,3,4'|

At prediction time, the following post-processing is done to calculate the probability of each rating based on the class prediction probabilities.

|Rating Output|Rating Probability Equation*|
|---|---|
|0|1-P('1')|
|1|P('1')-P('2')|
|2|P('2')-P('3')|
|3|P('3')-P('4')|
|4|P('4')|

* Negative values are clamped at 0

The rating with the highest probability is chosen as the predicted rating.

### Training and Model combinations
The overall strategy was to design an ensemble model using a variety of convolutional neural network architectures appled to differing image pre processing. The following commbinations were made, resulting in 8 models.

|Architectures |
|---|
|1 Resnet34|
|2 EfficientNet-B6|

|Image Proc|
|---|
|1 None|
|2 Blur based contrast|
|3 CLAHE|
|4 CLAHE on L |

This results in 8 different predictive combinations that were reviewed for high-scoring prediction on the validation set. 

#### Training
Training was done on a desktop PC with a 6 core CPU (Intel i7-8700) and 8GB GPU (RTX2070).  Training took about 48 hours.

For each model training was done in two phases.  In the first phase, only the final output layer was trained, leveraging the pretrained weights.  This training consisted of 10 epochs.  In the second phase the full network was trained over 20 epochs.

Example of the results of training Resnet34 with no image processing.

![resnet34train](https://github.com/filipmu/Kaggle-APTOS-2019-Blindness/blob/master/doc_images/training%20resnet34-none.png)

Validation set performance

![confusion](https://github.com/filipmu/Kaggle-APTOS-2019-Blindness/blob/master/doc_images/confusion%20resnet34-done.png)

### Model selection
Model selection was based on the best validation scores.  The final number of selected models for the ensemble was 7:
Resnet34-None, EfficientNet-B6-None, Resnet-34-Blur,Resnet34-CLAHE, EfficientNet-B6-CLAHE,Resnet34-CLAHEL, EfficientNet-B6-CLAHEL.

### Ensemble Model
To generate one prediction from the 7 models, the rating probabilities were summed across the models and the rating with maximum sum was chosen as the output.



Full notebook (via nbviewer): https://nbviewer.jupyter.org/github/filipmu/Kaggle-APTOS-2019-Blindness/blob/master/APTOS%202019%20Blindness%20v11%20-%20effnet-b6.ipynb
