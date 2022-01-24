# Google-Landmark-Recognition-2021
Have you ever gone through your vacation photos and asked yourself: What is the name of this temple I visited in China? Who created this monument I saw in France? Landmark recognition can help! This technology can predict landmark labels directly from image pixels, to help people better understand and organize their photo collections. This competition challenges Kagglers to build models that recognize the correct landmark (if any) in a dataset of challenging test images.

## Dataset
The dataset consists of 1580470 images of 81313 unique landmarks.

## Neural Network Architecture
<p align="center"><img src="https://github.com/NickKaparinos/Google-Landmark-Recognition-2021/blob/pytorch/Images/architecture.png" alt="drawing" width="400"/>
  
## Additive Angular Margin Loss (ArcFace)
### Overview
Additive Angular Margin Loss (ArcFace) is a state of art loss function used for image classification and face recognition. ArcFace has a clear geometric interpretation due to the exact correspondence to the geodesic distance on the hypersphere.
https://arxiv.org/abs/1801.07698
  
### ArcFace inference process
During inference, the features of the two images are normalised and the similarity is computed to determine if both pictures belong to the same class. The similarity between images is calculated using cosine similarity, which is a method used by search engines and can be calculated by the inner product of two normalised vectors.
  
### ArcFace versus Cross Entropy Loss
In a standard classification network, SoftMax and Categorical Cross-Entropy loss are usually used at the end of the network. SoftMax transforms numbers into probabilities. For each object, it gives a probability for each class that sums to 1. Once training is complete, the class with the highest probability is chosen. The Categorical Cross-Entropy loss calculates the difference between two distributions of probabilities and is minimized in the process of back-propagation during the training.
  
The drawback with SoftMax is that it does not produce a safety margin, which means that the borders are a bit blurry. We want the vectors of two images of the same person to be as similar as possible, and the vectors of two images of two different people to be as different as possible. That means we want to produce a margin, as SVM does.
  
<p align="center"><img src="https://github.com/NickKaparinos/Google-Landmark-Recognition-2021/blob/pytorch/Images/arcface_vs_softmax2.png" alt="drawing" width="600"/>
  
## Results
Using Stochastic Gradient Descend with only **3 epochs** of training, a validation accuracy of **25.5%** and a micro F1 score of **0.25** was achieved. Due to computing limitations, no further optimisation could be done. Future work could include further network architecture optimisation, larger image dimensions and more training epochs.
