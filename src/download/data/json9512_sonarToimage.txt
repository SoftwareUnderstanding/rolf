# sonarToimage
A vacation research scheme project. <br>

start date: 28 Nov 2019 <br>
end date: 12 Feb 2020

## Objective ##
To create a terrain for the virtual environment from sonar scans using GAN

## Work Progress ##
##### Data Preparation Phase #####
Created an intensity map of the sonar scans from the given dataset. <br>
The intensity map consists of (intensity value X number of angles) to create an image<br>
For example, if the intensity map is trying to show 30 degrees of the scan data. <br>
The dimensions will be (176, 15) where 176 is the maximum intensity value and 15 is 30 degrees / 2.<br>
(The scans are measured with an angle increment of 2 degrees between each scan).<br><br>
The CLAHE (Contrast-Limited Adaptive Histogram Equalization) filter was used to adjust the spotlight effect of the raw camera images. <br> 
Then, both the intensity map images and the filtered camera images were resized to the same size (64x64). <br><br>
After resizing the intensity map image and the raw camera image from the dataset to an ideal size for the network to train, <br>
the intensity map and the camera image are paired together for training. 

##### Model Setup & Training Phase #####
The pix2pix model from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/pix2pix/pix2pix.py was used as the basis for the code implementation for this network. <br>
Few Conv layers were removed from the Generator to suit the dataset.<br><br>
For the generator, the intensity map was given to produce an outcome that would be ideally similar to the camera images from the dataset.<br>
No data augmentation has been implemented in this model. <br>
<br>
**Generator Structure**
![ALT](sample/imgs/gen.PNG)

##### Test Phase #####
One point to consider is that the dataset itself only had camera images for angles that were directly below (90 deg) the autonomous vehicle.<br> Hence, only a limited amount of scan data had the ground truth (raw camera image) to be compared with.<br>
Majority of the scan data lacked ground truth. <br><br>
During the test phase, the model was able to produce images for scan data that lacked the ground truths.<br>
The result was not perfect, however, it was interesting to see the outcome.

## Result ##
The first row is the sonar intensity map<br>
second is the GAN generated image<br>
third is the original image<br>
<br>

After training 0 samples:<br>
![Alt](sample/imgs/0.png)
<br>
After training 5000 samples:<br>
![Alt](sample/imgs/5000.png)
<br>
After training 12000 samples:<br>
![Alt](sample/imgs/12000.png)
<br>
<br>
Final Image of GAN generated image tiles for all the sonar scans:<br>
![Alt](sample/imgs/image.png)

##### Limitations & Future Work #####
The major limitation for this project was the lack of data. The model would have been able to perform better if it were to have more data to be trained with. <br><br>
For the future, the project might be a fonudation that could redesign the way terrains are built in VR/AR environments. <br>
Currently, the machine learning part is separated from the VR part. However, the project could carry on and merge both parts together to visualize the final product.<br>
Ideally, it would be magnificent if the model could generate terrain in VR/AR as the data is being captured.

### Reference ###

###### Underwater cave sonar data near Spain:<br>
Link: https://cirs.udg.edu/caves-dataset/ <br>
Mallios, A.; Vidal, E.; Campos, R. & Carreras, M. <br>
Underwater caves sonar data set<br>
The International Journal of Robotics Research, 2017, 36, 1247-1251<br>
doi: 10.1177/0278364917732838<br>

###### pix2pix GAN pytorch implementation: <br>
Link: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/pix2pix/

###### pix2pix paper:<br>
Link: https://arxiv.org/abs/1611.07004<br>
Isola, P., Zhu, J. Y., Zhou, T., & Efros, A. A. (2017). Image-to-image translation with conditional adversarial networks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1125-1134).

### Weekly Log
|**Week**|**Tasks**                                                    |
|--------|---------------------------------------------------------------|
| 1 | - Getting familiar with Machine Learning/ Neural Network/ GANs<br>- Examining ROS data |
| 2 | - Tried to make laser scan to point cloud within the ROS functions<br>- However, the data seemed to be unorganized when transferring from one form to the other |
| 3 | - Trying to create an image directly from the raw laser scan data |
| 4 | - Customize the created image to sync with the camera information<br>- Reshape the synced image to ideal size for training |
| 5 | - Train the model and examine the output<br>- Recreate the pix2pix model to suit our purpose |
| 6 | - Create the model to suit our purpose<br>- Implement data augmentation techniques on the model<br>- Learn how to use the HPC |
| 7 | - Try to generate data for angles that are outside the camera frame |
| 8 | - Create an image of the full 360 sweep<br>- Apply data augmentation to avoid spotlight effect|
| 9 | - Apply ~~image vignette correction~~ CLAHE to make the outcome meaningful<br>- Find the correlation between the scan data input and the generated outcome |
| 10| - Optimize the code, Rewrite the code that were from other sources<br>- Crop the images to get rid of the borders |
| **Supervisors**| Dr. Ross Brown r.brown@qut.edu.au, Dr. Simon Denman s.denman@qut.edu.au |
