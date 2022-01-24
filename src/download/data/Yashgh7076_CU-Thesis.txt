# Clemson University MS Thesis - Segmentation and recognition of eating gestures from wrist-motion using deep learning, May 2020.

This is a repository for my CE Master's Thesis. The code is written in C (data processing) and Python (model design and implemenation). It is inspired by the U-Net model seen here https://arxiv.org/abs/1505.04597 and TensorFlow's own page on image segmentation seen here https://www.tensorflow.org/tutorials/images/segmentation.

Our novel idea is to treat an instant of recorded time same as a pixel in an image. This allows us to treat data recorded from inertial measurement unit (IMU) sensors as 2D images over time. For more information on the data please see http://cecas.clemson.edu/~ahoover/cafeteria/, and for a brief description of the project along with the link to download the document please see https://yashgh7076.github.io/projects.html. 

If you find the work useful please cite it as:

@mastersthesis{segmentation&recognition,
author = {{Y.Y.}~Luktuke},
type = {Master's Thesis},
title = {Segmentation and recognition of eating gestures from wrist motion using deep learning},
school = {Clemson University},
month = {May},
year = {2020}
}
