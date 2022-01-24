# IVP_Seminar_ws2020

**Contains the Code for the Group Project Topic 2: Image Classification**

## General information

Supervisor: Rakesh Rao Ramachandra Rao: rakesh-rao.ramachandra-rao@tu-ilmenau.de

Image classification refers to a process that can classify an image according to its visual content. 
For example, an image classification algorithm may be designed to tell if an image contains a cat or not. 
Many image classificaton algorithms, both classical and machine-learning-based, have been proposed in literature. 

**The main goals of this task are as follows:**
* Select 3 image classification algorithms of your choice
  * The choice should be based on an objective reason
  * These algorithms can either be classical or machine-learning-based
* Apply these 3 algorithms on the dataset that is provided 
* Analyse the results of the 3 classification algorithms

**submit a 4-page IEEE style paper (including references) until 04.02.2021 (11:59 PM) to Ashutosh Singla via email.**

## Current project status:
### Selection of Approaches
Selected three CNN-Approaches for image classification based on state-of-the-art benchmarks for CNNs trained on ImageNet database (TOP 1 and TOP 5 accuracy | different amount of training pictures)
  * NASNetLarge https://arxiv.org/abs/1707.07012
  * EfficientNetB7 https://arxiv.org/abs/1905.11946
  * FixEfficientNet-L2 https://github.com/facebookresearch/FixRes
* Further reading:
  * https://paperswithcode.com/task/image-classification
  * https://towardsdatascience.com/state-of-the-art-image-classification-algorithm-fixefficientnet-l2-98b93deeb04c 
  * https://www.analyticsvidhya.com/blog/2020/08/top-4-pre-trained-models-for-image-classification-with-python-code/ 
 
### Ground truth for data
* Did a ground truth on the 200 pictures https://cloud.tu-ilmenau.de/s/9ffWqt7G9oLziAk with up to 20 possilbe labels per image
* Selected one lable for each picture for a TOP 1 Accuracy calculation
 
### Working with high resolution images 

**DUE TO LIMITED PROCESSING POWER: UNTIL NOW ONLY USED IMAGES WITH SHRINKED RESOLUTION** </br>
https://cloud.tu-ilmenau.de/s/AKo8jqwEXLFs7Q3 </br>
Used the Image Resizer application with following options: </br>
custom width x height (450 x 300) | rotate and flip option: keep original | output setting **Retain the original format**

</br> **Identified possible approaches for working with high resolution images** 
* **Brute force approach**: Resizing to required dimension (Open CV) within the preprocessing for the specific cnn
* **PCA approach**: tdb

### Get the classification results

**NasNet Large (@Alex)**
- [x] Brute force approach: done
  * results in csv
  * statistical measurements in excel
- [ ] PCA approach: tbd

**EfficientNetB7 (tba)**
- [ ] Brute force approach: tba
- [ ] PCA approach: tbd

**FixEfficientNet-L2 (@Asad)**
- [ ] Brute force approach: tba
- [ ] PCA approach: tbd

## Possible next steps in the project 
* compare the results of the networks with ground truth 
  * **TOP 5 Acc with all available up to 20 gt-labels (no ranking)**
  * (tbd) TOP 5 Acc with 1 gt-label (label chosen by hand)
  * (tbd) TOP 1 Acc with all available up to 20 gt-labels (no ranking)
  * (tbd) TOP 1 Acc with 1 gt-label (label chosen by hand)
 
* (tbd) compare the influence of the pre-processing Resizing vs. PCA
 
* (tbd) Run the pre-processing with open cv or PCA on the full resolution images (difficult due to processing power)
 

 

