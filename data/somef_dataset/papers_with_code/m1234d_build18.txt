# WHAC


source devel/setup.bash
sudo chmod 666 /dev/ttyUSB0 (and /dev/ttyACM0)
urdf:
hector_mapping/mapping_launch.default
for laser and base link and odom to mapping

mapping:

roslaunch rplidar_ros rplidar.launch
roslaunch roboclaw_node roboclaw.launch
roslaunch hector_slam_launch tutorial.launch
rosrun teleop_twist_keyboard teleop_twist_keyboard.py

saving the map:
rosrun  map_server map_saver -f [map name here]

you can close everything now


localization:

launch lidar
launch roboclaw
launch hector_slam
rosrun map_server map_server [name of map.yaml] /map:=[any name here except for 'map']

in rviz: 
untick map (on the left panel)
click add, by topic, and click the new map (whatever you had named it)


## Components
1. LIDAR System (rplidar + our lidar code)
2. ZED Mini + ORB-SLAM2 (orb-slam2 + our slam code)
3. Motion Planning
4. Robot

## Code
In total there will be 6 ROS packages. Two for LIDAR, two for SLAM, one for motion planning and one for robot control.

## Run
After building,

For realsense:
Run slam: `roslaunch whac_slam whac_slam.launch`

For ZED:
Run ZED: `roslaunch zed_wrapper zed.launch`
Run slam: `roslaunch whac_slam whac_slam.launch`

## Important Links
ORB-SLAM2 Paper: https://arxiv.org/pdf/1610.06475.pdf

ZED SLAM implementation: https://github.com/yifenghuang/ZSLAM_TX2

ORB-SLAM2 code: https://github.com/raulmur/ORB_SLAM2
