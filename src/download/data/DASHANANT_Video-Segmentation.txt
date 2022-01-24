# Video-Semantic-Segmentation


Unet Architecture 
----------------------
![Unet](https://github.com/DASHANANT/Video-segmentation-using-Unet/blob/main/Unet%20Architecture%20.jpg)



why high computational training
-------------------
We need to train our model on video data which can take very long time also the storage requirement for Video data is more costly

Solution
------------------------------
*Transfer Learning*
----------------------------------
Decide to use transfer learning technique and we trained our model on IIIT Pet image dataset 
which contain both pet image and segmented pet image 
Now using that data we trained our model and decided to use it for video real time segmentation

*The results were amazing!!!!!* 

Here one output from webcam
=====================================
![](https://github.com/DASHANANT/Video-segmentation-using-Unet/blob/main/outpu.png)

Future Work
----------------------------
- To remove Background 
- To use augmented reality and use another background for real time video

References
=========
* Olaf Ronneberger, Philipp Fischer, & Thomas Brox. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation.
https://arxiv.org/abs/1505.04597
