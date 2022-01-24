DenseNet base Mask RCNN for Human Pose Estimation
-----------------------------------

The original code is from "https://github.com/matterport/Mask_RCNN" on Python 3, Keras, and TensorFlow. The code reproduce the work of "https://arxiv.org/abs/1703.06870" for human pose estimation.
This project aims to addressing the [issue#2][1]. 
When I start it, I refer to another project by [@RodrigoGantier][2] .

## Requirements
* Python 3.5+
* TensorFlow 1.4+
* Keras 2.0.8+
* Jupyter Notebook
* Numpy, skimage, scipy, Pillow, cython, h5py
# Getting Started
* [inference_humanpose.ipynb][5] shows how to predict the keypoint of human using my trained model. It randomly chooses a image from the validation set. You can download pre-trained COCO weights for human pose estimation (mask_rcnn_coco_humanpose.h5) from the releases page (https://github.com/Superlee506/Mask_RCNN_Humanpose/releases).
* [train_humanpose.ipynb][6] shows how to train the model step by step. You can also use "python train_humanpose.py" to  start training.
* [inspect_humanpose.ipynb][7] visulizes the proposal target keypoints to check it's validity. It also outputs some innner layers to help us debug the model.
* [demo_human_pose.ipynb][8] A new demo for images input from the "images" folder. [04-11-2018]
* [video_demo.py][9] A new demo for video input from camera.[04-11-2018]

![Original Image](https://github.com/lixiaolei1982/DenseNet-base-Mask-RCNN-for-Human-Pose-Estimation/blob/master/images/000000004134.jpg
)
![Pose Image](https://github.com/lixiaolei1982/DenseNet-base-Mask-RCNN-for-Human-Pose-Estimation/blob/master/images/untitled8.png)
![Mask Image](https://github.com/lixiaolei1982/DenseNet-base-Mask-RCNN-for-Human-Pose-Estimation/blob/master/images/untitled9.png)


  [1]: https://github.com/matterport/Mask_RCNN/issues/2
  [2]: https://github.com/RodrigoGantier/Mask_R_CNN_Keypoints
  [3]: https://github.com/matterport/Mask_RCNN
  [4]: https://github.com/RodrigoGantier/Mask_R_CNN_Keypoints/issues/3
  [5]: https://github.com/Superlee506/Mask_RCNN/blob/master/inference_humanpose.ipynb
  [6]: https://github.com/Superlee506/Mask_RCNN/blob/master/train_human_pose.ipynb
  [7]: https://github.com/Superlee506/Mask_RCNN/blob/master/inspect_humanpose.ipynb
  [8]: https://github.com/Superlee506/Mask_RCNN_Humanpose/blob/master/demo_human_pose.ipynb
  [9]: https://github.com/Superlee506/Mask_RCNN_Humanpose/blob/master/video_demo.py
  [10]: https://github.com/facebookresearch/Detectron/blob/master/lib/utils/keypoints.py
  [11]: https://github.com/QtSignalProcessing
  [12]: https://github.com/matterport/Mask_RCNN/issues/2
