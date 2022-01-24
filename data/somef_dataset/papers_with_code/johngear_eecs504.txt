# EECS504 Final Project
team:     Privacy Advocates

project:  Facial Anonymization in Video 

due:      December 14, 2020

## List of Materials used for reference
Residual NN for Images: https://arxiv.org/abs/1512.03385

Documentation for how OpenCV NN was trained: https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/how_to_train_face_detector.txt

SqueezeNet: https://arxiv.org/pdf/1602.07360.pdf   https://github.com/forresti/SqueezeNet  TF Implement: https://github.com/vonclites/squeezenet

EdgeNet is a small sized SqueezeNet-like architecture with FPGA implementation. Sort of useless other than proof of concept. Used on drones for edge computing. https://ieeexplore-ieee-org.proxy.lib.umich.edu/document/8617876

ZynqNet seems like a different variant of EdgeNet. https://arxiv.org/pdf/2005.06892.pdf

Dataset Used: WIDER Face: A Face Detection Benchmark (Yang, Shuo and Luo, Ping and Loy, Chen Change and Tang, Xiaoou, IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016



## From the submitted project proposal:

### Description 
Our project will process videos that contain human faces and return video with all facial features removed, either by performing a blur with randomized parameters, or by omitting facial pixels all together. We will likely do this by training a convolution neural network with a dataset used for video facial recognition, such as ‘Youtube Faces with Facial Keypoints’ found here. 

Existing video editing software has blurring functionality, but the user often has to select the features, and it’s unclear whether deblurring could reveal the identity after-the-fact. There are a few papers and similar projects available online that have demonstrated such work, such as this research paper, the following two articles, and the work of Terrance Boult and Walter Schierer.

If time and project complexity allow, an additional portion of the project could be examining feasibility of an on-device-algorithm that could be used on a camera so there was no back-door to deanonymize the data. 

### Demo
We hope to provide side-by-side video of before and after the algorithm runs on a variety of scenes containing people. It would be cool to implement it so that we could run it on live video, but achieving this level of efficiency with our methods may not be feasible. 
