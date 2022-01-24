Youtube Video: https://www.youtube.com/watch?v=QtfVU_17tac
Github: Too difficult to include weights and models.

Sources:
1. OpenPose: Original Paper by Zhe Cao, Gines Hidalgo, Tomas Simon, Shih-En Wei, Yaser Sheikh https://arxiv.org/abs/1812.08008 https://github.com/CMU-Perceptual-Computing-Lab/openpose
2.  YOLOv3: Original paper by Joseph Redmon and Ali Farhadi. https://arxiv.org/abs/1804.02767
3. YOLOv3: PyTorch implementation by Ultralytics LLC. https://github.com/ultralytics/yolov3
4. ROS: https://www.ros.org/
5. Kinetic4RPiZero: @nomumu https://github.com/nomumu/Kinetic4RPiZero/
6. DFRobot IO Expansion Board Driver: https://github.com/DFRobot/DFRobot_RaspberryPi_Expansion_Board
7. raspicam_node: https://github.com/UbiquityRobotics/raspicam_node

Computer Prerequisites (Base Program):
1. OS: Tested on Ubuntu 18.04, should work on Ubuntu >= 14.04
2. Graphics Card: Nvidia GPU (CPU can work but is slow)
3. Python 3
4. OpenCV: Tested on 4.2.0
5. PyTorch: Tested on 1.5
6. CUDA: Tested on 10.2
7. OpenPose

Computer Prerequistes: (MINI-Q)
1. ROS: Tested on Melodic

Car Prerequisites: (MINI-Q)
1. MINI-Z compatible RC Car
2. Raspberry Pi Zero W
3. Raspberry Pi Camera
4. DFRobot IO Expansion HAT for Pi Zero/Zero W
5. OS: Raspbian 2018-04-19
6. DFRobot Raspberry Pi Expansion Board
7. ROS: Tested on Kinetic https://github.com/nomumu/Kinetic4RPiZero/



Basic Guide:
1. Install Nvidia Driver and CUDA: https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=runfilelocal
2. Either in Anaconda or through pip, install PyTorch: https://pytorch.org/
3. Install OpenPose: https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation.md
4. Install OpenCV: There are many ways to do this. For example, ROS full installation comes with OpenCV.
5. Clone YOLOv3 and download weights: https://github.com/ultralytics/yolov3
6. You should be able to run pedestrian_awareness/main.py at this point

How to make the 1:28 car (This is a complicated process and I(Eric) plan to release a guide on how exactly I made it):
1. High-end Kyosho MINI-Z model or other 1:28 RC cars that allow you to access the ESC and servo directly through PWM (Base MINI-Z needs modification.)
2. 5V BEC on ESC
3. Raspberry Pi Zero W
4. Raspberry Pi Zero compatible camera
5. DFRobot IO Expansion Hat for Pi Zero/ Zero W (Needs to flip a protection diode or remove it)

ROS Guide:
1. Install ROS Melodic or Kinetic on your Ubuntu: https://www.ros.org/install/
2. Install ROS Kinetic on Raspbian: https://github.com/nomumu/Kinetic4RPiZero/
3. Make catkin workspace on Raspberry Pi that includes all the libraries and files in minizbot_catkin_src
4. Set up ROS multi-machine and run roscore on ROS master
5. SSH into the Raspberry Pi and launch the miniz node by roslaunch minizbot miniz.launch
6. Run pedestrian_awareness/bot.py on ROS master
