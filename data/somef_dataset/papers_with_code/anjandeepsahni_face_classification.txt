# Face Classification and Verification

## Introduction

Given an image of a person’s face, the task of classifying the ID of the face is known as **face classification**. Whereas the problem of determining whether two face images are of the same person is known as **face verification** and this has several important applications. This mini-project uses convolutional neural networks (CNNs) to design an end-to-end system for face classification and face verification.

## Face Classification
Input to the system is a face image and it predicts the ID of the face. The true face image ID is expected to be present in the training data. In this way, the network will be doing an N-way classification to get the prediction.

## Face Verification
Input to the system is a trial, that is, a pair of face images that may or may not belong to the same person. Given a trial, the system will output a numeric score that quantifies how similar the faces of the two images appear to be. The system uses the final convolution layer as an embedding which represents important features from a person. It uses cosine similarity to assign a confidence score to two images. A higher score indicates higher confidence that the two images belong to one and the same person.

## Preprocessing
The following preprocssing methods are not implemented and are left for future work.
- **Face Detection**: Face detection is the automatic process for detection of human faces in digital images. This will ensure that the model you are training only sees images of humans and any noise in the images is deleted.
- **Face Alignment**: Face alignment is the automatic process of identifying the geometric structure of human faces in digital images. Given the location and size of a face, it automatically determines the shape of the face components such as eyes and nose. Given the location of the face in the image, images can be cropped to include only the human faces, without any background noise. This will also reduce noise for the model training.

## Models

Trimmed down version of ResNet50 and MobileNetV2 are supported. Model ensembling of three different MobileNetV2 implementations and a single ResNet50 implementation provides the best results. Some of my observations are listed below:

- I found that MobileNetV2 is much faster then ResNet50 and is a more suitable architecture for smaller datasets. ResNet50 takes too much time to converge, and might end up giving better results if trained for a long duration. I accounted the slower convergence to the fact that the network will take time to learn that a lot of the filters are useless.
- I used the “Top 3 Highest Validation Accuracy” and “Top 2 Lowest Validation Loss” for 4 different architectures, that is, three MobileNetV2 and one ResNet50. So total 20 models were used for final prediction in classification as well as verification. I found model ensemble to improve my results by 7-8% on validation set.

## System Evaluation
Given similarity scores for many trials, some threshold score is needed to actually accept or reject pairs as same-person pairs (i.e., when the similarity score is above the threshold) or different-person pairs (i.e., when the score is below the threshold), respectively. For any given threshold, there are four conditions on the results:

- False Positive: Some percentage of the different-person pairs will be accepted.
- False Negative: Some percentage of the same-person pairs will be rejected.
- True Negative: Some percentage of the different-person pairs will be rejected.
- True Positive: Some percentage of the same-person pairs will be accepted.

The Receiver Operating Characteristic (ROC) curve is created by plotting the True Positive Rate (TPR) against the False Positive Rate (FPR) at various threshold settings. The Area Under the Curve (AUC) for the ROC curve is equal to the probability that a classifier will rank a randomly chosen similar pair (images of same people) higher than a randomly chosen dissimilar one (images from two different people) (assuming ’similar’ ranks higher than ’dissimilar’ in terms of similarity scores). This AUC is used as the evaluation metric for face verification.

## Results
Custom/private dataset was used for this task. The results achieved were at par with the expectations. Please note that the test accuracy is much lower than the validation accuracy due to imbalance in the dataset. Best results from independent MobileNetV2 and ResNet50 models are as shown below.

| Architecture | Validation Accuracy | Validation Loss | Test Accuracy |
|--------------|:-------------------:|:---------------:|:-------------:|
| MobileNetV2  |        69.2 %       |       1.4       |     60.6 %    |
| ResNet50     |        65.4 %       |       1.6       |     56.2 %    |

With model ensembling as explained in the **Models** section, below results were achieved:

| Task                | Test Accuracy |
|---------------------|:-------------:|
| Face Classification |     68.8 %    |
| Face Verification   |   92.7 (AUC)  |

## References
- **MobileNetV2: Inverted Residuals and Linear Bottlenecks**: https://arxiv.org/pdf/1801.04381.pdf
- **Deep Residual Learning for Image Recognition**: https://arxiv.org/pdf/1512.03385.pdf
