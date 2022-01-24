# Purpose 

The purpose of the repository is to gain an understanding of PointNet and try it out for myself for the purpose of my Final Year Project (FYP). My FYP will look into 3D Object Detection Algorithms and PointNet is the basis feature extractor for many 3D Object Detection Algorithms which is why PointNet is of interest to me. The code repository was taken from [Fei Xia](https://github.com/fxia22/pointnet.pytorch). I have investigated the details of the algorithms and reported it here (and in my FYP). 

# PointNet

Point clouds are unordered set of points which prevents usage of CNN object detection algorithm as CNN assumes inputs to be in an ordered data structure. Thus, point cloud is usually transformed into an ordered representation to provide the benefit of applying convolutions. There are different ways to represent LiDAR point cloud before feeding it into the network for training if we want to use 2D Convolution operations or 2D Object Detection network such as putting it in the form of range images %LaserNet
, bird-eye-view feature maps and pillars.
% need read into voxelnet and second

In 2017, [a seminal paper PointNet](http://stanford.edu/~rqi/pointnet/) was released showing that it is possible to take raw point cloud data directly to do classification and segmentation while ensuring permutation invariant \citep{qi2016pointnet}. Before PointNet, point cloud data must be transformed into other ordered representation such as voxels which had disadvantage of corrupting the data and being computationally expensive. It is worth investigating into PointNet as numerous 3D object detection and segmentation algorithms uses PointNet as the fundamental building blocks to the network. The official implementation of PointNet can be found [HERE](https://github.com/charlesq34/pointnet).


PointNet is a uniﬁed architecture that directly takes point clouds as input and outputs either class labels or per point segment/part labels of the input. The PointNet architecture as seen in the Figure below. 

![img](images/pointnet.PNG)

The first component is a point cloud encoder that learns to encode sparse point cloud data into a dense feature vector. The PointNet encoder model is composed of three main modules :


1. Shared Multilayer perceptron (MLP)
2. Max Pooling Layer
3. Joint Alignment Network (Input transform and Feature transform)


The shared MLP layer is implemented using a series of 1D convolution, batch normalization, and ReLU operations. The 1D convolution operation is configured such that the weights are shared across the input point cloud to gather information regarding the interaction between points. It is useless to passed the weight of a single point isolated from other points as there are no useful information provided. The shared MLP layers can then learn local information of the point cloud and provide a local feature vector. Effectively the network learns a set of optimization functions/criteria that select interesting or informative points of the point cloud and encode the reason for their selection. The ﬁnal MLP layers of the network aggregate these learnt optimal values into the global descriptor for the entire shape and feed it into the max pooling layer. We will be able to extract out the global information of the point cloud from local feature vectors and provide a global feature vector from the output of the max pooling layer.

The max pooling layer is very important because it is a symmetric function which ensures that the network achieves permutation invariance given that point cloud inputs are unordered. A symmetric function is a function that outputs the same values regardless of the order of the inputs given to it. It is worth noting that we are free to choose other symmetric functions instead of max pooling.  


A data-dependent spatial transformer network attempts to canonicalize (standardise) the data before the PointNet processes them can be added to improve the results of the network. This is done by the input transform and feature transform model in the PointNet architecture. The transform model learns to predict a transformation matrix using a mini-network (T-Net) and directly apply this transformation to align the inputs to a canonical space. The T-Net resembles the big network and is composed by basic modules of shared MLP ,max pooling and fully connected layers as seen in Figure below. 

![img](images/tnet.png)


The T-Net Network regresses to find the transformation matrix that provides invariance to orientation changes such as rotation and translation by setting up the loss function according to the property of an orthogonal matrix where <img src="https://render.githubusercontent.com/render/math?math=A^T = A^{-1}, AA^T = I">. Thus, the loss function is:

<img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle++L+%3D+%7C%7C+I+-+AA%5ET+%7C%7C%5E2_F" 
alt=" L = || I - AA^T ||^2_F">

We can observed that the loss function is set up to minimise the loss so that the A matrix gets closer to that of an orthogonal matrix, where A is the 3x3 matrix of the input transform or 64x64 of the feature transform that is applied to every point cloud input through matrix multiplication.

The <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%7C%7C.%7C%7C_%7BF%7D" 
alt="||.||_{F}"> is the Frobenius norm. The Frobenius norm decomposes the matrix into a singular value by reshaping the matrix in a single column vector and taking the square root of the inner product of the column vector (Euclidean norm), the equation for Frobenius norm is given by:

<p align="center">
  <img src="https://github.com/timothylimyl/PointNet-Pytorch/blob/master/images/equation_img.PNG" alt="Sublime's custom image"/>
</p>


The Frobenius norm is a very important step as the loss function has to be singular value. The loss function from the T-Net is then combined with the classification/segmentation loss.


The second component of PointNet after the encoder is the classifier that predicts the categorical class scores for all points or it can also be used for segmentation by concatenating the local feature and global feature vector together. Any multi-class loss function such as negative log likelihood can be used.

There a few desirable properties of PointNet which provided it with superior results. It processes the raw points directly preventing information loss from operations like voxelization or projection. It also scales linearly with the number of input points. The total number of parameters needed by PointNet is way lower than 3D Convolutions. For example, MVCNN (3D Convolution method) uses 60 Million parameters while PointNet only need 3.5 Million parameters.

The issue with set up of the PointNet architecture is that it does not capture local structures of the points as PointNet only learns the spatial encoding of each individual point and aggregates all the point features to a global feature vector. PointNet lacks the ability to capture local context at different scales. It is very useful if the network can learn from local context like a Convolutional Neural Network (CNN) instead of just encoding each point individually. Learning from local context at different scales helps abstract different local patterns which allows for better generalizability. For example, the first few layers of the CNN extract simple representation such as corners,edges and spots and layers after it builds on top of these local context to form more complex representations. The author of PointNet proposed PointNet++ to fix this issue. PointNet++ partitions the point clouds into different sets and apply PointNet to learn local features. The architecture of PointNet++ can be seen in Figure below.


![img](images/pointnet_plusplus.JPG)


[PointNet++](https://arxiv.org/abs/1706.02413) architecture builds a hierarchical grouping of points and progressively extract features off larger groupings along the hierarchy. Each level of hierarchy provides with different abstraction through Set Abstraction as seen in Figure \ref{fig:pointnet++}. Set abstraction is made of three layers:

1. Sampling layer
2. Grouping layer
3. PointNet layer 

The Sampling layer chooses a set of points from the input points to define the centroids of local region, the grouping layer takes the centroids of the local region and group all of the neighbouring points together to construct local region sets. Each local region set is then encoded into feature vectors using PointNet. By getting feature vectors from the sets of local region, the architecture can get different local context such as edges and curves which is one of the reasons for the performance of PointNet++ being better than PointNet. For example in ModelNet40 shape classification, PointNet++ has an accuracy of 91.9\% in comparison to PointNet with 89.2\%.

In the interest of my final year project, I am nterested in usage of PointNet networks for 3D Object Detection. Usage of PointNet for 3D Object Detection is similar to how CNN Architectures are used for 2D Object Detection, which is as the base network for feature extraction. The features extracted/learnt can be connected end-to-end to a regression head for predicting 3D Bounding Boxes. There are many different 3D Object Detection that uses PointNet such as  Frustum PointNet, PointFusion and PointPillar that I will explore in detail.


