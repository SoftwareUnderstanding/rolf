# CV
This is a directory for Computer Vision.
These codes are available on the developing environment of google colaboratory.
Reference material are https://japan-medical-ai.github.io/medical-ai-course-materials/index.html  (written in Japanese).
Details of what I implemented are below:


(1) Supervised Segmentation of Medical Images. 
   
I used the data of MRI images of hearts (256*256 sizes) and the segmented data of the same sizes.

The number of the data is just 282, but the scores such as Pixcel Accurecy and Intersection over Union are quite high, 
using the proper architecture of Fully Convolutional Network or U_Net



(2) AutoEncoder using U_Net

the architecture is almost same to (1). 

Just only a AutoEncoder is a little bit meaningless. However, this arcitecture can be utilized in the part of Generator of Gan. For example, this paper https://arxiv.org/pdf/1611.07004.pdf introduce conditional GAN using U_Net, for the task of Image to Image, such as reconstruction of Monochrome Image to Colorful Image.  
