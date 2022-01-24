# ImageSegmentation_Unet

Anomalies in subsurface cause diffraction hyperbolas in GPR profile. An attempt for recognizing those patterns using deep learning technology was conducted on a synthetic GPR image. For each pixel in a GPR image, I would like to know the category (hyperbola or not) it belongs to. A U-Net architecture was applied for this segmentation problem. 

U-Net was originally developed for biomedical image segmentation (Ronneberger et al 2015) and has later on been successfully used for seismic interpretation (Li et al 2019). A U-Net consists of a contracting path which follows the typical architecture of a convolutional network, and a symmetric expanding path that enables precise localization. I t has advantage in requiring less training images and yielding more precise segmentations (Ronneberger et al 2015).

![alt text](https://github.com/GeoDQ/ImageSegmentation_Unet/blob/main/Picture1.png)
![alt text](https://github.com/GeoDQ/ImageSegmentation_Unet/blob/main/Picture2.png)

### References
##### Ronneberger et al., 2015. U Net: Convolutional Networks for biomedical image segmentation . https://arxiv.org/pdf/1505.04597.pdf
##### Li et al., 2019. Seismic fault detection using an encoder decoder convolutional neural network with a small training set. Journal of Geophysics and Engineering, 16, 175-189
