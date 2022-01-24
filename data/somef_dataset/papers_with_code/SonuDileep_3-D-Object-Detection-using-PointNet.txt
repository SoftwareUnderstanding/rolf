## PointNet: *Deep Learning on Point Sets for 3D Classification and Segmentation*
### Introduction
This is a work by Charles R. Qi, Hao Su, Kaichun Mo, Leonidas J. Guibas from Stanford University.You can find the link to their paper here[https://arxiv.org/abs/1612.00593]. Their original work is done using tensorflow1.x and few other packages which are deprecated in newer version. So, I implemented their model using tensorflow 2.0 and python 3.7. Shown below is an example of 3D point cloud objects in ModelNet40 Dataset. You need to do few more steps to train /test the model.

Step_1 -> Install h5py<br>
          sudo apt-get install libhdf5-dev<br>
          sudo pip install h5py<br>
Step_2 -> Download the ModelNet40 dataset and copy the files to data folder. Type the following command in the terminal and unzip the file.<br>
<b>wget --no-check-certificate -P "pointnet/data" https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_v0.zip</b><br>
 Step_3 -> In terminal use "python train.py". For evaluation run, "python evaluate.py --visu"<br>
<b> I have not included the weight files here cause of it's size. But if you need the weight files you can pull a request, I can share via my drive </b>
 
![3D point cloud vase](https://github.com/SonuDileep/3-D-Object-Detection-using-PointNet/blob/master/vase.jpg)

