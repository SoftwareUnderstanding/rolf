# ShuffledVideoSorter
This project attempts to reorder a set of images from a shuffled video using deep learning.

The flow of the project is quite simple: read the images, apply background substraction, apply an already trained CNN to obtain the feature map, select the first image and find the next one using the L2 distance (or cosine) with respect to the feature map, do the same until you run out of images. This way you end up with two parts of the video, possible in reverse, which can be reordered manually. The feature map was obtained using the pre-trained MobileNetV2 architecture: https://arxiv.org/abs/1801.04381 .

To run the pipeline run main.py with two arguments "input folder" and "output folder".

The first 12 sorted images given by the pipeline:

<p float="left">
  <img src="sorted_images/image_0.jpg" width="250" />
  <img src="sorted_images/image_1.jpg" width="250" /> 
  <img src="sorted_images/image_2.jpg" width="250" />
  <img src="sorted_images/image_3.jpg" width="250" />
  <img src="sorted_images/image_4.jpg" width="250" />
  <img src="sorted_images/image_5.jpg" width="250" />
  <img src="sorted_images/image_6.jpg" width="250" /> 
  <img src="sorted_images/image_7.jpg" width="250" />
  <img src="sorted_images/image_8.jpg" width="250" />
  <img src="sorted_images/image_9.jpg" width="250" />
  <img src="sorted_images/image_10.jpg" width="250" />
  <img src="sorted_images/image_11.jpg" width="250" />
</p>

