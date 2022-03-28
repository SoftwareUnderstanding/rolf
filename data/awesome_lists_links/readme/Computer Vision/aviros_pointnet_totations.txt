## *Semi-Supervised Classification By Predicting Image Rotations In Point Cloud*

### Introduction
In this work, 3D image features are learned in a self-supervised way by training pointNet network to recognize the 3d rotation that been applied to an input image. Those learned features are being used for classification task learned on top of this base network.

In this repository I uploaded code and data.

### Installation

You may need to install <a href="https://www.tensorflow.org/get_started/os_setup" target="_blank">TensorFlow</a>. You may also need to install h5py.

To install h5py for Python:
```bash
sudo apt-get install libhdf5-dev
sudo pip install h5py
```

### Usage
To train a complete model to classify point clouds sampled from 3D shapes run the following bash script:

	!chmod 777 train.sh
     run.sh
     
The training split into two parts and can be run with different parameters (see code): 

self-suprevised task predicting randomly rotating images:
	
	python trainRotation.py --model_save_path=fourRotations --rotation_list=[0,3,5,6]
(rotation_list default value is [0,3,5,6]. It is list of numbers in between 0-7 represents the pi/2 rotation upon the axis. i.e, 3 represents X not rotate, Y rotate, Z rotate)

Then, Classifier is learned on top of previous network. Exampled usage:
	
	python trainClasiffiers.py \
	--model_save_path=fc3_stop_gradient_4rotations \
	--model_restore_path=fourRotations \
	--fc_layers_number=3 \
	--freeze_weights='True'

(Note: 'model_restore_path'  have to be consistent with the same name of 'model_save_path' parameter from trainRotation.py, as it use the trained parameters of previous network at initialization)

Log files and network parameters will be saved to `log` folder in default. Point clouds of <a href="http://modelnet.cs.princeton.edu/" target="_blank">ModelNet40</a> models in HDF5 files will be automatically downloaded (416MB) to the data folder. Each point cloud contains 2048 points uniformly sampled from a shape surface. Each cloud is zero-mean and normalized into an unit sphere. There are also text files in `data/modelnet40_ply_hdf5_2048` specifying the ids of shapes in h5 files.

To see HELP for the training script:

We can use TensorBoard to view the network architecture and monitor the training progress.

    tensorboard --logdir log
