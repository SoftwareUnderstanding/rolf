# Transformer Networks

Repository with the implementation in PyTorch of visual attention mechanisms called Spatial Transformer
Networks (STN) and CoordConv layers for MNIST classification.

In this repository we provide two different networks in PyTorch:
- A CNN with Spatial Transformer Networks, designed for making the original network more robust to transformations in input data, e.g., rotations, traslations, etc. 
- A modification of the first network with the addition of CoordConv layers to the Localization Network (LN) of the STN. This type of layers are meant to provide Conv layers with information about the coordinates of the input images. The LN has the task of obtaining a affine transformation matrix from input images, and the CoordConv layers
have shown to improve accuracy in that type of tasks.


-------------------------------------------------------------------------------------------------------------------------------

## Spatial Transformer Networks

Based on the paper: "Spatial transformer networks", Max Jaderberg et al., Advances in neural information processing systems, 2015, vol. 28, p. 2017-2025. https://arxiv.org/abs/1506.02025 

![Header](images/STN.PNG)

Spatial transformer networks are a generalization of differentiable attention to any spatial transformation. STNs allow a neural network to perform spatial manipulation on the input data within the network to enhance the geometric invariance of the model. 

CNNs are not invariant to rotation and scale and more general affine transformations. In STNs the localization network is a regular CNN which regresses the transformation parameters. The transformation network learns automatically the spatial transformations that enhances the global accuracy on a specific dataset.

STNs can be simply inserted into existing convolutional architectures without any extra training supervision or modification to the optimization process.

-------------------------------------------------------------------------------------------------------------------------------

# CoordConv Layers

In this repository also an implementation of a STN with the addition of CoordConv layers is provided. 

![Example](images/CoordConv.PNG)

Convolutions present a generic inability to transform spatial representations between two different types: from a dense Cartesian representation to a sparse, pixel-based representation or in the opposite direction. CoordConv layers were designed to solve this limitation modifying the traditional convolutional layers by adding information about the coordinates of the input images to the input tensor. The CoordConv layer is designed to be used a substitute of the regular Conv2D layer.

CoordConv layers are presented in the paper:  "An intriguing failing of convolutional neural networks and the coordconv solution", Rosanne Liu et al., arXiv preprint arXiv:1807.03247 (2018). 
https://arxiv.org/pdf/1807.03247.pdf

-------------------------------------------------------------------------------------------------------------------------------

## How to use the code:

-- Assuming you have an environment with all software dependencies solved:

1) Download or clone the repository to a local folder:

       git clone "https://github.com/uam-biometrics/Spatial-Transformer-Networks.git"
      
2) Files and descriptions:

- models.py: definition of the backbone CNN’s architecture.
- coord_conv.py: definition of the CoordConv layers.
- utils.py: definition of various functions used during training and testing.
- spatial_transformer_tutorial.py: script for training the network with STNs and Conv2D layers. Based on the tutorial in https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html. 
- spatial_transformer_nets_with_coord_convs.py: script for training the network with STNs and CoordConv layers.
- evaluate_models.py: script for testing the pretrained models with the MNIST dataset.
- stn_classic.pt and stn_coordconv.pt: PyTorch trained models.


-- Using the models for replicating our results for MNIST classification:
  
1) You have to run the evaluate_models.py script : it loads the already trained models and evaluates them on the test partition of the MNIST dataset. 

-------------------------------------------------------------------------------------------------------------------------------

## Results for MNIST classification:

Results have been obtained after training the models during 20 epochs.

![Example](images/results.PNG)

![Example](./images/MNIST_example.png)

Examples of image warping for a test batch of the MNIST dataset. CoordConv layers obtain better results in most of the cases. For example, the two upper-left number 1s are more vertical in the CoordConv case than when using only STN with Conv2D layers.


