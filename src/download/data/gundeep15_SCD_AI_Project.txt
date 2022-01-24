Implementing deep learning to segment and classify sickle cells using the morphological cell-to-cell variation. 
 
Biophysical deformability of sickle rbcs is directly visible in their phase-contrast microscopic images as differences in cell shape, size, and concavity. The dynamic deformability of sickle cells is further correlated with the adhesion characteristics of sickle cells. These morphological differences can therefore be used to characterize the intrapatient cell-cell adhesion. 

2 different AI-based networks are developed and trained to 
1. Identify and segment adhered cells using a UNet-based architecture (https://arxiv.org/abs/1505.04597):  
2. Classify the segmented cells using a ResNet-block architecture (https://arxiv.org/abs/1512.03385):

A typical image of our complete bed of microvasculature is shown below. The image is (15000 X 5250) px in size and contain cells on the order of 10<sup>1</sup>-10<sup>3</sup> cells with diverse morphological characteristics, thus making the problem highly challenging. 

<img src="36 Hypoxia.jpg" width="480">

An application for generalizing the cell segmentation and classification algorithms we have used in this project are underway. 
