# MaskRCNN-Demo
This repo is about Mask RCNN (paper: https://arxiv.org/abs/1703.06870) with human-keypoint for human pose estimation. This repo relies on FangYang970206's [repo](https://github.com/FangYang970206/MaskRCNN-Keypoint-Demo).

# Requirements
* Python 3.4+
* TensorFlow 1.3+
* keras 2.0.8+
* numpy, skimage, scipy, Pillow, cython, h5py
* cocoapi: `pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"`
* cv2: `pip install opencv-python`

# Run in notebook [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Krupal09/MaskRCNN-Demo/blob/master/demo.ipynb)
You can open the notebook in google colab, the environment can prepare in one cell and has free GPU.Just enjoy!

# Run in local
first, clone this repo,
```bash
$ git clone https://github.com/Krupal09/MaskRCNN-Demo
```
then,
```bash
$ cd MaskRCNN-Keypoint-Demo
```
download the **pre-trained model**([dropbox](https://www.dropbox.com/s/5ctrg3br94srrx9/mask_rcnn_coco.h5)) in the MaskRCNN-Keypoint-Demo folder.

finally :
```bash
$ python main.py --image path/to/image
```
