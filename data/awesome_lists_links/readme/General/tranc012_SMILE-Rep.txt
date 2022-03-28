# SMILE Lab University of Florida
# Momentum Contrast Learning on the RSNA Dataset
## Representation Learning 
## HiPerGator - A100 

This is a very minimal repo. The primary code is sourced from Momentum Contrast for Unsupervised Visual Representation Learning ( https://arxiv.org/abs/1911.05722 ) with minor changes for compatiability and hyperparameter adapations. Due to the large size, I cannot provide the saved model weights nor the data directly here. Note: the reported problem is related to the computational speed, not the accuracy/performance. 

Code Summary 
 
Data: The data used was from a medical CT image Kaggle dataset (RSNA Intracranial Hemorrhage Detection https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/overview ).  We split this data by percentages, and end up with the corresponding number of images: unsupervised data (564601), training data (150560), and testing data (37640). To show the effectiveness of the method on reduced labeled images, 20% of the training data was used giving 30112 images for the benchmark experiment on the A100. This is a multi-label classification problem from 6 classes. 

Model Architecture: The MoCo paper utilizes a ResNet-50 style encoder. A fully connected layer is used to train a linear classifier after the unsupervised training stage. 

RSNA Image Size: 224 x 224 

CIFAR10 Image Size: 32 x 32 

Batch Size: 128 

Epochs: For the benchmark task of computational time, we will use 100. This is not the optimal number for the best accuracy report though. 

Learning Rate: 0.03 followed by a lr decay rate. 

Framework: PyTorch (on A100: version 1.6.0) 

Optimizer: Adam 

Loss Functions: (Unsupervised: Noise Contrastive Estimation), (Supervised: Cross-Entropy)

Evaluation Metric: AUC

# MoCo_Downstream and Some Instructions

1) The only code you will need is MoCo_Downstream
2) You need to change some directories. I share a screenshot of what needs to be changed.

![3epoch](https://user-images.githubusercontent.com/57649485/90648071-17ca3e00-e207-11ea-9c0e-89aa876dd1ab.PNG)

3) It will print out the time reports per epoch and a final breakdown time report at the end.

![2Epoch](https://user-images.githubusercontent.com/57649485/90647992-fa956f80-e206-11ea-9ee2-62cff2db715c.PNG)

General Information: 

These saved weights are used for a supervised downstream task (classification). In this supervised stage, we fine-tune the entire model along with a linear classifier, initialized by the saved weights from the previous unsupervised step. As mentioned before, in the HiPerGator implementation, 20% of the training data was used along with the test set, evaluated at every epoch. For the benchmark task, the speed was examined on 100 epochs. 

For some note in the code MoCo_downstream, there are some true-false flags that can be passed in order to specify certain model designs: other considerations are Resnet-50 training (in which the unsupervised weights are not passed, giving a basic Resnet-50 supervised training), and MoCo-Freeze (in which the supervised weights are passed but the model is frozen, training only the linear classifier). 

Personal results from various GPU configurations:

![Figure1](https://user-images.githubusercontent.com/57649485/88983920-45703700-d29a-11ea-94e2-1398620c3de9.PNG)

I have made a slight change to formatting from the original run. The biggest change I made from the previous figure is by calculating the validation time on its own rather than calculating the time for the whole epoch. 
