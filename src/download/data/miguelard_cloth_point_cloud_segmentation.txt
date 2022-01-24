# Cloth Point Cloud Segmentation
Real-time multi-cloth point cloud segmentation ROS package. Object detection, image and point cloud processing are combined for segmenting cloth-like objects (dishcloths, towels, rags...) from point clouds. The implementation is based on [YOLOv4](https://arxiv.org/abs/2004.10934?), [GrabCut](https://www.microsoft.com/en-us/research/wp-content/uploads/2004/08/siggraph04-grabcut.pdf) and [Color-based region growing segmentation](https://www.isprs.org/proceedings/xxxviii/3-w8/papers/p65.pdf).

**Authors**: Miguel Arduengo, Ce Xu Zheng, Adrià Colomé and Carme Torras.

**Affiliation**: All the authors are with [Institut de Robòtica i Informàtica Industrial, CSIC-UPC (IRI)](http://www.iri.upc.edu/), Barcelona. 

<p align="center">
	<img src=".github/media/fig1.gif" width=860>
</p>

## Contents
- [Algorithm Overview](#algorithm-overview)
	+ [Cloth Detection](#1-cloth-detection)
	+ [Color Image Segmentation](#2-color-image-segmentation)
	+ [Point Cloud Segmentation](#3-point-cloud-segmentation)
- [Installation](#installation)
	+ [Dependencies](#dependencies)
	+ [Building](#building)
- [Basic Usage](#basic-usage)
- [ROS Implementation](#ros-implementation)
	+ [Subscribed Topics](#subscribed-topics)
	+ [Published Topics](#subscribed-topics)

## Algorithm Overview
The cloth point cloud segmentation algorithm requires a color image and the corresponding point cloud. The following steps are then performed sequentially:

### 1. Cloth Detection
Cloth-like object detection is performed on the color image using a custom [YOLOv4](https://arxiv.org/abs/2004.10934?) model trained specifically for this purpose. You only look once (YOLO) is a state-of-the-art, real-time object detection system that provides a list of object categories present in the image along with an axis-aligned bounding box indicating the position and scale of every instance of each object category. In this way, just a small region of interest around the cloth can be extracted for further processing.

<p align="center">
	<img src=".github/media/fig2.gif" width=275 height=200>
	<img src=".github/media/fig3.gif" width=275 height=200>
	<img src=".github/media/fig4.gif" width=275 height=200>
</p>

### 2. Color Image Segmentation 
Segmenting the cloth requires a far more granular understanding of the object in the image. For classifying the pixels within the bounding boxes between those that belong to the cloth and those that belong to the background the [GrabCut](https://www.microsoft.com/en-us/research/wp-content/uploads/2004/08/siggraph04-grabcut.pdf) algorithm is used. Starting with the bounding box around the cloth to be segmented, pixels are divided according to the estimated color distribution of the target object and that of the background. For enhacing the algorithm performance a small axis-aligned elliptical region that certainly belongs to the cloth, whose size is relative to the bounding box dimensions, is defined. Also, morphological operations are applied for refining the resulting mask provided by GrabCut. Note that, since segmentation is color-based, the best performance is obtained for cloths with a uniform color that contrasts with the background.

<p align="center">
	<img src=".github/media/fig5.png" width=410 height=170>
	<img src=".github/media/fig6.png" width=410 height=170>
</p>

### 3. Point Cloud Segmentation
The algorithm requires the point cloud to be organized, that is, the points that are adjacent in the cloud structure also correspond to adjacent pixels in the color image. Then, the mask obtained in the previous step can also be used to filter the point cloud since the points are arranged following the same structure than the pixels in the color image. Additionally, taking advantage of the spatial information enconded in the point cloud, the segmentation is refined using the [color-based region growing segmentation](https://www.isprs.org/proceedings/xxxviii/3-w8/papers/p65.pdf) method. The purpose of the said algorithm is to merge the points that are close enough in terms of both distance and color, dividing the input cloud into a series of clusters. In this way, the cloth points are merged together into a single cluster, sepparating them from points that might have not been filtered in the previous step. Note that again, the best performance is obtained for cloths with a uniform color.

<p align="center">
	<img src=".github/media/fig7.png" width=850>
</p>


## Installation

### Dependencies

The cloth point cloud segmentation package depends on the following software:
* [Robot Operating System (ROS)](http://wiki.ros.org/ROS/Installation): Software libraries and tools for robot applications.
* [OpenCV](https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html) (>= 4.x): Open source computer vision and machine learning software library.
* [Point Cloud Library (PCL)](https://pointclouds.org/downloads/): Standalone, large scale, open project for 2D/3D image and point cloud processing.
* [YOLOv4 requirements](https://github.com/AlexeyAB/darknet#requirements): State-of-the-art, real-time object detection system.

### Building
In order to install the cloth point cloud segmentation package, first clone the latest version of this repository, including all the submodules, in your catkin workspace.

```
  cd ~/catkin_workspace/src
  git clone --recurse-submodules https://github.com/MiguelARD/cloth_point_cloud_segmentation.git
  cd ..
```

Then compile the package using ROS. The first time it might take some minutes.

```
  catkin_make
```

Alternatively, you can also compile the package using the [catkin command line tools](https://catkin-tools.readthedocs.io/en/latest/index.html#).

```
  catkin build
```

## Basic Usage
Running the cloth point cloud segmentation ROS package is very simple. First, you have to download the [YOLOv4 network weights](https://drive.google.com/file/d/1ua9XE0xd5pX8GwNdo98NDojhlTGP6-Dg/view?usp=sharing) and place it in the `cloth_segmentation/yolo_network_config/weights` folder. 

```
cd ~/catkin_ws/src/cloth_point_cloud_segmentation/cloth_segmentation/yolo_network_config/weights

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1ua9XE0xd5pX8GwNdo98NDojhlTGP6-Dg' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1ua9XE0xd5pX8GwNdo98NDojhlTGP6-Dg" -O yolo_cloth.weights && rm -rf /tmp/cookies.txt
```

Once the weights are in the corresponding folder, specify the topics where your RGB-D camera is publishing the color image, the point cloud and the raw image in the [params.yaml](cloth_segmentation/config/params.yaml) file from the `cloth_segmentation/config` folder.

```yaml
# Camera topics
subscribers:

  # Color image
  rgb_reading:
  	topic: /your_color_image_topic
  	
  # Point cloud
  point_cloud_reading: 
  	topic: /your_point_cloud_topic
  	
  # Raw image
  camera_reading:
    	topic: /your_raw_image_topic
```


Finally, once your camera is running in ROS, source your catkin workspace and launch the package. 

```
cd ~/catkin_ws
source devel/setup.bash
roslaunch cloth_segmentation cloth_segmentation.launch
```

If everything is working correctly you should see something similar to the figure below. A [rviz](http://wiki.ros.org/rviz) window, a display for the YOLOv4 detections and a sepparate terminal for printing the detections' information will appear. Note that in order to visualize it in rviz correctly you have to specify your `fixed_frame` and the topics where your camera is publishing the color image and the point cloud. These parameters can be set directly on the rviz interface in the places highlighted in red. 

<p align="center">
	<img src=".github/media/fig8.png" width=850>
</p>


## ROS Implementation

The ROS implementation of the multi-cloth point cloud segmentation package consists on a single node [`cloth_segmentation`](cloth_segmentation/src/cloth_segmentation.cpp), which depends on the package [`darknet_ros`](https://github.com/tom13133/darknet_ros/tree/9f4843298576cbda949909f546d8757095cb9032) to obtain the YOLOv4 detections. 

### Subscribed Topics

+ **`/darknet_ros/bounding_boxes`** (type [`darknet_ros_msgs::BoundingBoxes`](https://github.com/leggedrobotics/darknet_ros/blob/master/darknet_ros_msgs/msg/BoundingBoxes.msg))

    Array of bounding boxes that gives information of the position and size of the bounding box corresponding to each cloth in the image in pixel coordinates.

+ **`/camera/rgb/image_color`** (type [`sensor_msgs::Image`](http://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/Image.html))

    Color image provided by the RGB-D camera. This topic can be specified in the [params.yaml](cloth_segmentation/config/params.yaml) file from the `cloth_segmentation/config` folder.
    
+ **`/camera/depth_registered/points`** (type [`sensor_msgs::PointCloud2`](http://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/PointCloud2.html))

    Point cloud provided by the RGB-D camera. This topic can be specified in the [params.yaml](cloth_segmentation/config/params.yaml) file from the `cloth_segmentation/config` folder.


### Published Topics

+ **`/cloth_segmentation/cloth_image`** (type [`sensor_msgs::Image`](http://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/Image.html))

    Color image that includes only the pixels corresponding to the segmented cloths after performing [step 2](#2-color-image-segmentation) of the algorithm.

+ **`/cloth_segmentation/cloth_pointcloud`** (type [`sensor_msgs::PointCloud2`](http://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/PointCloud2.html))

    Point cloud that includes only the points corresponding to the segmented cloths after performing [step 3](#3-point-cloud-segmentation) of the algorithm.
