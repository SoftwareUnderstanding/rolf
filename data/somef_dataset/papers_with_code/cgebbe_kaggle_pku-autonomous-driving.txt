# About

This is the code for my first kaggle competiton https://www.kaggle.com/c/pku-autonomous-driving 

![img](README.assets/img.png)


###  Problem
- Given a single image, detect all cars in the image and estimate their pose(translation + rotation)

- The prediction is evaluated according to the mean Average Precision (mAP) metric with different thresholds for the pose error.
  - Explanation of AP, see https://sanchom.wordpress.com/tag/average-precision/
  - Explanation of mAP, see https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173
- The dataset consisted of approx. 5k images with 60k labels and was provided by Baidu and Peking University

### Solution approach

- Use the Center-Net implementation, see https://arxiv.org/pdf/1904.07850.pdf
  - It detects the (2D) center point of objects as a binary mask (using binary cross-entropy loss / log loss) or as a heatmap (using focal loss, see paper)
  - All other properties (six pose variables) are regressed using an L1 loss
- Post-process the estimated pose
  - Use the fact that the estimated xyz-coords should overlap with the estimated uv-coords of center point
  - Use the fact that most center points lie on a flat plane, i.e. y_CCS is approximately linearly dependent of (z_CCS, x_CCS)
- See my detailed progress in [log.md](log.md)

### Results

- My predictions scored place 145 out of 864 (top 17%), see https://www.kaggle.com/gebbissimo/competitions
- I did learn a lot of valuable lessons about kaggle, free GPU use (provided by kaggle and google colab) as well as strategies of how to improve CNN models


# FAQ

### How to install

- Use the default python3 libraries installed in kaggle and google colab notebooks 
- Download the dataset from https://www.kaggle.com/c/pku-autonomous-driving 
- Adapt the parameters in the file `params.yaml`, most importantly the paths to the dataset

### How to run 

Simply run  `python3 main.py`
