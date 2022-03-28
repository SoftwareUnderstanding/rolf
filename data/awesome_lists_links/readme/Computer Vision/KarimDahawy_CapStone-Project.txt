# Capstone Project

## Description
--------------------------------------------------------------
This is the project repo for the final project of the Udacity Self-Driving Car Nanodegree-Capstone Project: Programming a Real Self-Driving Car. 

## Project Member:
--------------------------------------------------------------
This project is done by: *Karim Dahawy* (karim.dahawy@valeo.com)

[//]: # (Image References)

[image1]: ./imgs/Capstone_Ros_Graph.png
[image2]: ./imgs/Waypoint_Updater_Node.png
[image3]: ./imgs/DBW_Node.png
[image4]: ./imgs/Traffic_Light_Detection_Node.png
[image5]: ./imgs/move.png
[image6]: ./imgs/Decelerate.png
[image7]: ./imgs/stop.png
[image8]: ./imgs/Accelerate.png

## Introduction:
--------------------------------------------------------------
In order to design a fully autonomous Vehicle the following techniques have been used:
    
    1. Waypoint Following techniques
    2. Control
    3. Traffic Light Detection and Classification
    
The Waypoint Following technique would take information from the traffic light detection and classification with the current waypoints in order to update the target velocities for each waypoint based on this information.

For Control part, I have designed a drive-by-wire (dbw) node that could take the target linear and angular velocities and publish commands for the throttle, brake, and steering of the car. 

For Traffic Light Detection and classification, I have designed a classification node that would take the current waypoints of the car and an image taken from the car and determine if the closest traffic light was red or green.
 
![alt text][image1]

## Project Details:
--------------------------------------------------------------
### 1. Waypoint Following techniques:
-------------------------------------

This is considered as a ROS Node that listens or subscribes to (/base_waypoint), (/current_pose), and (/traffic_waypoint) topics in order to generate or publishes (/final_waypoint).

![alt text][image2]

This technique is excuted based on the following:
    
 1. Generating the final waypoints to make the vehicle moves on straight lines.
 2. Use the Controller part in order to control throttle, steering and brake actions of the Autonomous Vehicle.
 3. Integrating the traffic light detection and classification, so this node subscribes to (/traffic_waypoint) topic.
 4. The (/final_waypoint) is updated based on the traffic light color:
    * if RED, the velocity of the vehicle decelerates through the future waypoints.
    * if GREEN, the velocity accelerates till the Maximum allowable speed through future waypoints.
      
### 2. Control:
---------------

This is considered as a ROS Node that subscribes to (/twist_cmd), (/current_velocity), and (/dbw_enabled) topics in order to publishes (/vehicle/steering_cmd), (/vehicle/throttle_cmd), and (/vehicle/brake_cmd).

![alt text][image3]

This Part is responsible to control the vehicles (throttle, steering, and brake) action commands.
A PID controller is built with parameters (KP = 0.3, KI = 0.1, KD = 0). This part is called Drive by Wire (dbw) which can be defined as having electric control signal for the main control actions of the vehicle. The brake value is functional of the vehicle mass and the wheel radius calculating the vehcile Torque.
      

### 3. Traffic Light Detection and Classification:
-------------------------------------------------

This is considered as a ROS Node that subscribes to (/base_waypoints), (/image_color), and (/current_pose) in order to publishes (/traffic_waypoints).

![alt text][image4]

The Purpose of this part is to build a deep learning model to detect the position of the traffic light in the image sent by Carla Simulator, then classify its color if it is RED or GREEN. 

Using Bosch traffic light data (https://hci.iwr.uni-heidelberg.de/node/6132), I was able to train a simple classification network (less inference time) that takes the image and output the traffic light color. 

A fine-tuned MobileNet (https://arxiv.org/pdf/1704.04861.pdf) is offered a good balance between efficiency and accuracy. I have depended on the information of stop line locations, so we decided not to use an object detection, and instead classify entire images as conraining very simply: RED, YELLOW, or GREEN traffic light.


### Vehicle Performance on Unity Simulator

The vehicle is oving Normally on the Simulator:

![alt text][image5]

The vehicle is able to decelerate if the traffic light is RED:

![alt text][image6]

The vehicle stops while the traffic light is RED: 

![alt text][image7]

The vehicle is able to accelerate if the traffic light is GREEN:

![alt text][image8]

## Installation

Please use **one** of the two installation options, either native **or** docker installation.

### Native Installation

* Be sure that your workstation is running Ubuntu 16.04 Xenial Xerus or Ubuntu 14.04 Trusty Tahir. [Ubuntu downloads can be found here](https://www.ubuntu.com/download/desktop).
* If using a Virtual Machine to install Ubuntu, use the following configuration as minimum:
  * 2 CPU
  * 2 GB system memory
  * 25 GB of free hard drive space

  The Udacity provided virtual machine has ROS and Dataspeed DBW already installed, so you can skip the next two steps if you are using this.

* Follow these instructions to install ROS
  * [ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu) if you have Ubuntu 16.04.
  * [ROS Indigo](http://wiki.ros.org/indigo/Installation/Ubuntu) if you have Ubuntu 14.04.
* [Dataspeed DBW](https://bitbucket.org/DataspeedInc/dbw_mkz_ros)
  * Use this option to install the SDK on a workstation that already has ROS installed: [One Line SDK Install (binary)](https://bitbucket.org/DataspeedInc/dbw_mkz_ros/src/81e63fcc335d7b64139d7482017d6a97b405e250/ROS_SETUP.md?fileviewer=file-view-default)
* Download the [Udacity Simulator](https://github.com/udacity/CarND-Capstone/releases).

### Docker Installation
[Install Docker](https://docs.docker.com/engine/installation/)

Build the docker container
```bash
docker build . -t capstone
```

Run the docker file
```bash
docker run -p 4567:4567 -v $PWD:/capstone -v /tmp/log:/root/.ros/ --rm -it capstone
```

### Port Forwarding
To set up port forwarding, please refer to the "uWebSocketIO Starter Guide" found in the classroom (see Extended Kalman Filter Project lesson).

### Usage

1. Clone the project repository
```bash
git clone https://github.com/udacity/CarND-Capstone.git
```

2. Install python dependencies
```bash
cd CarND-Capstone
pip install -r requirements.txt
```
3. Make and run styx
```bash
cd ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
```
4. Run the simulator

### Real world testing
1. Download [training bag](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic_light_bag_file.zip) that was recorded on the Udacity self-driving car.
2. Unzip the file
```bash
unzip traffic_light_bag_file.zip
```
3. Play the bag file
```bash
rosbag play -l traffic_light_bag_file/traffic_light_training.bag
```
4. Launch your project in site mode
```bash
cd CarND-Capstone/ros
roslaunch launch/site.launch
```
5. Confirm that traffic light detection works on real life images

### Other library/driver information
Outside of `requirements.txt`, here is information on other driver/library versions used in the simulator and Carla:

Specific to these libraries, the simulator grader and Carla use the following:

|        | Simulator | Carla  |
| :-----------: |:-------------:| :-----:|
| Nvidia driver | 384.130 | 384.130 |
| CUDA | 8.0.61 | 8.0.61 |
| cuDNN | 6.0.21 | 6.0.21 |
| TensorRT | N/A | N/A |
| OpenCV | 3.2.0-dev | 2.4.8 |
| OpenMP | N/A | N/A |

We are working on a fix to line up the OpenCV versions between the two.
