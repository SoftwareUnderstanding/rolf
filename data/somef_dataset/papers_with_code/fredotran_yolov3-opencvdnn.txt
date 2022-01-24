# Using OpenCV DNN combined with YoloV3 and YoloV3-tiny for simple object detections tasks.

![plot](results/imgs/detect_img3.jpg)

## Requirements.

* [Python3 or higher](https://www.python.org/downloads/)
* [OpenCV 4.0 or higher](https://opencv.org/releases/) 
* [CUDA 11.0 or higher](https://developer.nvidia.com/cuda-toolkit-archive) 
* [cuDNN 7.0 or higher](https://developer.nvidia.com/rdp/cudnn-archive) 
* [GPU with CudaCompiler](https://en.wikipedia.org/wiki/CUDA#GPUs_supported)
* Windows OS (I haven't tried on other OS)
* Lot of patience (but worth it)

You can watch tutorial steps to download and build [OpenCV with GPU back-end](https://medium.com/analytics-vidhya/build-opencv-from-source-with-cuda-for-gpu-access-on-windows-5cd0ce2b9b37), as we will need [OpenCV DNN](https://docs.opencv.org/master/d2/d58/tutorial_table_of_content_dnn.html) module for this work.

---

## Notebooks.

You'll find all the instructions in the three different notebooks which are basically the same, the only changes occur on the detection device or type of input frames. There are three different object detections tasks I would like to highlight there, for objects detections in images, videos and webcam (cameras). This code was inspired by some coders on github and Kaggle, they allow me to start my first project of Computer Vision using OpenCV DNN and YoloV3. 

All the functions are stored in these notebooks and ready to work :

* [Notebook for images](https://github.com/fredotran/yolov3-opencvdnn/blob/main/yolov3_on_images-tests.ipynb)
* [Notebook for videos](https://github.com/fredotran/yolov3-opencvdnn/blob/main/yolov3_on_videos-tests.ipynb)
* [Notebook for webcam/cameras](https://github.com/fredotran/yolov3-opencvdnn/blob/main/yolov3_on_camera-tests.ipynb)

---

## YoloV3 cfg and weights.

I based this repository on the YOLOv3 algorithm : https://pjreddie.com/darknet/yolo/ .
The weights and cfg files were downloaded on the official website, I used YOLOv3-320 (320x320) and YOLOv3-tiny. The weights and cfg files can be checked in the **[cfg](https://github.com/fredotran/yolov3-opencvdnn/tree/main/cfg)** and **[models](https://github.com/fredotran/yolov3-opencvdnn/tree/main/models)** folders.

---

## Dataset.

The YOLOv3-320 and YOLOv3-tiny were trained on the MSCOCO Dataset, you'll find the dataset listing names and bouding boxes with COCO format in **[data](https://github.com/fredotran/yolov3-opencvdnn/tree/main/data)** folder.

---

## Images and videos inputs.

You'll be able to find images and videos for test in the folders **[images](https://github.com/fredotran/yolov3-opencvdnn/tree/main/images)** and **[videos](https://github.com/fredotran/yolov3-opencvdnn/tree/main/videos)**, the code is already setup to execute directly in those directories.

---

## About

This code is not the most complicated and is certainly not the most detailed one but, it was the easiest one for me to start to learn about using OpenCV DNN and YOLOv3 for object detections and I think it might be useful for beginners to start there.
In the near future, I'll try to upgrade the code for custom detections and combined my findings in this : https://github.com/fredotran/traffic-sign-recognition with OpenCV DNN and YOLOv3 for traffic signs detections.

---

## Source 

***YOLOv3: An Incremental Improvement***, Redmon, Joseph and Farhadi, Ali, **arXiv**, 2018 : http://arxiv.org/abs/1804.02767

https://www.murtazahassan.com/
