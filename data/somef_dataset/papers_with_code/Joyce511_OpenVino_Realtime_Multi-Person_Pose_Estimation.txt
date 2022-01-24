# Realtime Multi-Person Pose Estimation
## 0.Introduction 
Developed by Zhe Cao, Tomas Simon, Shih-En Wei and Yaser Sheikh, this realtime multi-person pose estimation model utilizes a bottom-up approach, without using any person detector. 
This repo contains the Intel OpenVino implementation of this model.
CVPR'17 paper: 
https://arxiv.org/abs/1611.08050

Original Github repo: 
https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation

Caffe Model: 
http://posefs1.perception.cs.cmu.edu/Users/ZheCao/pose_iter_440000.caffemodel

Prototxt:
https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation/blob/master/model/_trained_COCO/pose_deploy.prototxt

## 1.Prerequisite
- Intel OpenVino Toolkit 2018 R2
- Jupyter Notebook
- Open CV2
- Python 3.5 or higher
- Numpy
- Scipy
- Matplotlib
- clDNN for GPU mode
## 2. Usage
- Download model files:
pose_iter_440000.bin 
https://drive.google.com/file/d/1vC6AORvo6oYJ0ub34GZ8ujTwgYQcSUkY/view?usp=sharing
Pose_iter_440000.caffemodel
https://drive.google.com/file/d/12KZUKdlZppRNvNw5eTsSrA_dI6jXCo7N/view?usp=sharing

- Place them under <OpenVino_Realtime_Multi-Person_Pose_Estimation>/Realtime_Multi-Person_Pose_Estimation/model/_trained_COCO
```
$ cd <OpenVino_Realtime_Multi-Person_Pose_Estimation>/Realtime_Multi-Person_Pose_Estimation/testing/python
$ jupyter notebook
```
- Open pose_estimation_openvino_picture.ipynb / pose_estimation_openvino_video_CPU.ipynb / pose_estimation_openvino_video_GPU.ipynb

- On the upper handlebar, click Cell > Run All
## 3. Performance
### Single-image performance

#### Test image
![alt text](https://github.com/Joyce511/OpenVino_Realtime_Multi-Person_Pose_Estimation/blob/master/Realtime_Multi-Person_Pose_Estimation/testing/sample_image/ballet.jpg)
#### Image size
- 620x465 px
#### Performance
- OpenVino w/ converted .xml model: ~700ms
- Caffe w/ original .caffemodel model: ~12000ms

- For further comparison, refer to pose_estimation_openvino_picture.ipynb and pose_estimation_caffe_picture.ipynb
### Video-input performance
#### Video size
- 320x240
#### Performance
- OpenVino w/ converted .xml mode, CPU model: ~ 1 frame/sec
- For further information, refer to pose_estimation_openvino_video_CPU.ipynb and pose_estimation_openvino_video_GPU.ipynb

## 4.Sample Results
![alt text](https://github.com/Joyce511/OpenVino_Realtime_Multi-Person_Pose_Estimation/blob/master/img/result1.png)
![alt text](https://github.com/Joyce511/OpenVino_Realtime_Multi-Person_Pose_Estimation/blob/master/img/result2.png)
