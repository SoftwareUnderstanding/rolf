# eyeSRGAN #

This project consists on a Tensorflow 2.2.0 implementation of an SRGAN model based on paper (https://arxiv.org/abs/1609.04802) applied to eye resolution in order to improve gaze estimation techniques by reducing the computing requirements or by improving estiamtion accuracy.

### What is this repository for? ###
The purpose of this repository is, in one hand, to show an example of how to implement an SRGAN model using one of the most recent versions of Tensorflow and, on the other hand, to show a possible application of super resolution techniques. This project was developed as my final master's degree project which was presented on July the 3rd at the Public University of Navarre. The project was completed with honors.

### Results ###
- Visual results.

![alt text](https://github.com/alvarobasi/eyeSRGAN/blob/master/images/RESULTS.png)

- Gaze estimation results. ([Paper](https://openaccess.thecvf.com/content_ICCVW_2019/papers/OpenEDS/Porta_U2Eyes_A_Binocular_Dataset_for_Eye_Tracking_and_Gaze_Estimation_ICCVW_2019_paper.pdf))

![alt text](https://github.com/alvarobasi/eyeSRGAN/blob/master/images/all_models_3.png)

- Pupil center estimation using an SDM algorithm ([Paper](https://openaccess.thecvf.com/content_ICCVW_2019/html/OpenEDS/Porta_U2Eyes_A_Binocular_Dataset_for_Eye_Tracking_and_Gaze_Estimation_ICCVW_2019_paper.html)).

Comparison between original and SR-generated images using an SRGAN model.
![alt text](https://github.com/alvarobasi/eyeSRGAN/blob/master/images/User18_comparison.png)
Comparison between original and SR-generated images using an ResNet-MSE model.
![alt text](https://github.com/alvarobasi/eyeSRGAN/blob/master/images/User18_comparison_mse.png)
Comparison between original and SR-generated images using a bicubic algorithm.
![alt text](https://github.com/alvarobasi/eyeSRGAN/blob/master/images/User18_comparison_bic.png)
