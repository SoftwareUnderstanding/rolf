# Dissertation_Image_segmentation
## Introduction of the model

This model is a semantic image segmentation model, which assigns label to each pixel of an image to partition different objects into segments. Semantic image segmentation means that there is no distinguishing between different plants, e.g. plant_1 and plant_2. They are all classified as plants as a whole. There exists models which can distinguish different objects of the same category. However, in this case, this function does not provide too much helps.

The whole model is composed of two parts, namely backbone part and classifier part. The backbone part has been pre-trained on a large dataset (may exceed over 1 million of training examples) to have a generalizing ability, while the classifier part is untrained, and should be fine-tuned based on specific tasks. This structure takes the advantage of transfer learning ability of deep learning models, which means that a well trained model can perform well in different similar tasks.

The main metric used to measure the performance of the model is Intersection over Union (IoU), which is 

![](https://github.com/sdyy6211/Dissertation_Image_segmentation/blob/master/gitpic/IOU.png)
|:--:| 
| *Visual explanation of IoU* |


## The aim of this model

This model is used to automatically segment each object within crowdsourced images, which are unstructured data that is considered as difficult to be processed automatically. The practical application of the model is to automatically detect and monitor the changes of historical sites from those unstructured image data overtime. In this case, the aim is to detect and monitor the growth of the plant on the wall of Bothwell castle. 

## Process

### Prepare input images from crowdsourced with the following image as an example
![](https://github.com/sdyy6211/Dissertation_Image_segmentation/blob/master/gitpic/original_image.jpg)
|:--:| 
| *An example of training image* |


### Hand labelling the image using an online tool (https://app.labelbox.com/)
![](https://github.com/sdyy6211/Dissertation_Image_segmentation/blob/master/gitpic/working.PNG)
|:--:| 
| *The interface of labelling tool* |
### Obtain the label for training examples (overall and local region)

![](https://github.com/sdyy6211/Dissertation_Image_segmentation/blob/master/gitpic/label.png)
|:--:| 
| *The label of overall classes for the first model* |

![](https://github.com/sdyy6211/Dissertation_Image_segmentation/blob/master/gitpic/segmentated_label.png)
|:--:| 
| *The label of binary classes for the second model to refine prediction* |

This process involves downloading masks from the Labelbox. These codes correspond to the file of *segmentation_data_processing.ipynb*

### Train the two models (the first model with 8 classes and a second model for binary classes) using the original images and labels

Deeplab v3+ with a backbone model of resnet101 is used in this project, implemented by https://github.com/jfzhang95/pytorch-deeplab-xception using PyTorch.

For details of training, I freeze the parameters of the backbone model, and only trained the deeplab head parameters with epoch number of 100 and learning rate of 0.01 for the first model, and epoch number of 100 and same learning rate for the second model. The optimizer is Adam (https://arxiv.org/abs/1412.6980) and the loss function is cross entropy loss with 8 classes for the first model and 2 classes for the second model. A scheduling stepdown of learning rate is applied to both models, which means the learning rate will reduce to its 1/10 every epoch of 50 (for the first model) or 33 (for the second model). This is used for the optimizer to better find the minimum point of the loss function.

### Results of overall segmentation

![](https://github.com/sdyy6211/Dissertation_Image_segmentation/blob/master/gitpic/predicted.png)
|:--:| 
| *Segmented objects* |

![](https://github.com/sdyy6211/Dissertation_Image_segmentation/blob/master/gitpic/superimpose.png)
|:--:| 
| *comparison of label and segmentation results, with an Intersection over Union (IoU) of 0.82966 for all classes and 0.56250 for plants* |

### Reverse selection of the bounding box of the detected objects

Since I would like to crop the area of plants, and further refine them using the second model, I need to obtain the coordinates of a bounding box of the segmented objects based on its maximum and minimum vertical and horizontal coordinates. This is achieved by using DBSCAN of sklearn (https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html) in *get_region_v2* of *utils_d.py*, which is an unsupervised clustering algorithm that automatically partitions disjoint objects. Therefore, distinguishing objects within a class can be partitioned to be drawn an individual bounding box. Once the coordinates of each object are determined, bounding boxes of each disjoint object can be drawn as shown by the following figure.

![](https://github.com/sdyy6211/Dissertation_Image_segmentation/blob/master/gitpic/fig7_annotations_with_rect.png)
|:--:| 
| *bounding box of each object* |

### Refined results

After having the bounding box, I can crop the plant from the whole image, and feed it into the second model to refine the prediction. The refined prediction is shown as follow.

![](https://github.com/sdyy6211/Dissertation_Image_segmentation/blob/master/gitpic/compare.png)
|:--:| 
| *selecting interested local area and refining predictions in the area* |

![](https://github.com/sdyy6211/Dissertation_Image_segmentation/blob/master/gitpic/re1com.png)
|:--:| 
| *comparison of label and segmentation results in the local refined area, with an Intersection over Union (IoU) of 0.80745, increased from 0.56250* |

### Image registration

Since the area of a plant in a picture is significantly affected by the angel of the picture taken, it would be better to register two images with different angels based on the window in the image using an affine transformation. The registration has three steps.

The first step is to find the most similar image from the training set. The similarity is measured by the *cv2.matchTemplate*

The second step is to detect the position of the window using the first model combined with a reverse selection of the bounding box of the window, which is similar to the detection of plants.

The third step is to select the area of window and make affine transformation on the whole image. The selection of area of window is based on traditional computer vision toolkits in opencv rather than deep learning.

![](https://github.com/sdyy6211/Dissertation_Image_segmentation/blob/master/gitpic/rect.png)
|:--:| 
| *The process of image registration based on the window* |

![](https://github.com/sdyy6211/Dissertation_Image_segmentation/blob/master/gitpic/register_outfalse.png)
|:--:| 
| *Final result* |

The final IoU of the two models on the test set are approximately 62% and 53% for the first and second model respectively.
