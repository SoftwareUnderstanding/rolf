# Frustum++

## Description
This is a half-semester project, an ongoing attempt to extend the original work by:
  1. Using LIDAR point cloud data instead of RGB-D data
  2. Using bird's eye view representation of LIDAR data to generate additional 3D bounding box proposals from the top view

The below figure shows the *proposed modification* to the original work. The components (at the top) connected by red arrows are what we are going to add.

![proposed_architecture](https://github.com/witignite/Frustum-PointNets/blob/master/doc/Fig_3_ProposedArchitecture.PNG)

The purpose is to experiment on using the bird's eye view LIDAR data to help improving the performance in some extreme condition, such as low light or occlusion, where it may be difficult to detect the 2D bounding boxes in the image (and hence no 3D object will be detected). By using the bird's eye view LIDAR data, we expect to obtain additional 3D box proposals, which will be combined with the original proposals to improve the performance.

## Dataset
- Because the KITTI dataset may be shared with multiple projects, I put the data outside of this project folder.

- To run the code, download KITTI 3D object detection <a href="http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d">dataset</a> and organize the folders as follows (replace `~/DataSet/KITTI/object_detect_3d/` with your preferred choice):

  ```
  ~/DataSet/KITTI/object_detect_3d/
      training/
          calib/
          image_2/
          label_2/ 
          velodyne/

      testing/
          calib/
          image_2/
          velodyne/
  ```
  Then set `KITTI_DATA_PATH` in `kitti/kitti_object.py` and `kitti/prepare_data.py` to your preferred path:

  ``` python
  KITTI_DATA_PATH = '~/DataSet/KITTI/object_detect_3d/'
  ```

## Library Dependencies
The following environment is used during development. (It is still in an experiment; not guaranteed to work. Refer to the original repository for a safe installation.)
- Python 3.6.9
- OpenCV 4.1.0
- TensorFlow 1.12.0
- NumPy 1.17.2 (This generates a lot of `FutureWarning`, may want to use `NumPy` < 1.17.0 during the installation.) To suppress the `FutureWarning`:
  ```python
  import warnings
  warnings.filterwarnings('ignore',category=FutureWarning)
  import tensorflow as tf
  ```
- mayavi 4.6.2
- Boost 1.77.0

## Notes about Installation
There are changes in some files that I made specifically for my own environment. Please refer to the original `README.md` <a href="https://github.com/witignite/Frustum-PointNets/blob/master/doc/">here</a> at "Installation" section for more details.
- In `3d_interpolation`, `grouping`, and `sampling` folder under `models/tf_ops` directory, change the `TensorFlow` and `CUDA` path in each `tf_*_compile.sh` file to point to libraries in your environment.
- Change `Boost` path in `train/kitti_eval/compile.sh`

## Notes about Running
- Change the `KITTI_DATA_PATH` as described above in the "Dataset" section.
- All of the scripts in `scripts` directory are modified to run in one specific environment. Change the variables, paths, and compier flags accordingly.
- The following script(s) are added:
  ```
  command_test_pretrained_v2.sh
  ```

<br/>

***

### Original Work
I decided to move the `README.md` of the original work to <a href="https://github.com/witignite/Frustum-PointNets/blob/master/doc/">here</a> to reduce the clutter on this page.
