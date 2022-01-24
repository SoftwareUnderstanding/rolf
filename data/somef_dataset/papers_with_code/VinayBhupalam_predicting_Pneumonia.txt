
Deep learning model for classification of Chest Xrays to detect Pneumonia 

### Dataset  <br/>
Downloaded from Kaggle paultimothymooney/chest-xray-pneumonia<br/>

### Deep Learning Framework: <br/>
Python Keras library<br/>

## Deep learning Architecture:<br/>
->Residual network is used with Batch Normalization after addition.<br/>
->Existing neural network is modified to contain multiple paths within the same residual block[ResNext Architecture].<br/>
->Refer research paper  https://arxiv.org/abs/1611.05431<br/>
  
## Data preprocessing:<br/>
->All the input images are read as grayscale images<br/>
->Resized to dimension of 256x256<br/>
->Input images are augmented by rotating the images by 40 degrees.Selection of input images as a candidate for Augmentaion is random.<br/>


## Optimization method:<br/>
->Gradient Descent with momentum<br/>
->Learning rate is gradually decayed.<br/>

## Loss function:<br/>
->Categorical Cross Entropy.<br/>

## Optimizing metrics<br/>
->AUC [Area under ROC Curve]<br/>






