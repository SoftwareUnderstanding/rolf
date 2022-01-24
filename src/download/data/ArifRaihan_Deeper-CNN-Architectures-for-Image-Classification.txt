# Deeper-CNN-Architectures-for-Image-Classification
As part of a coursework for the module Deep learning for computer vision, I had to research about some of the famous CNN architectures that helped make groundbreaking improvements in the way models are designed and trained now. After the 
significant advancement by AlexNet (2012) ,the most common approaches after that focused on improving by increasing the depth and the width of the networks. But the way of just 
experimenting with width and depth came with its major drawbacks – making networks more prone to overfitting and the expensive use of computational resources. I implemented and 
experimented with the architectures of VGG11 and GoogleNet by training and testing on the commonly used datasets MNIST and CIFAR-10. <b>The experimental results are discussed in details in the experimental results file in this repository </b>.

The paper for VGG16 focused on increasing the depth of the network and used smaller filter which improved the performance. They proposed 5 variants of the network with increasing parameters for each of them. The number of parameters increase with the
increasing depth and width of the network. I experimented with the smallest variant of this network (VGG-11)

The paper for InceptionNet aimed at solving the  very issue of utilizing computational resources by carefully designing a network with increasing number of layers while keeping the computational budget constant. They made use of inception modules stacked on top of each other, convolutional neural networks for
dimensionality reduction and extra classifiers to help in backpropagation. The name of the network that was submitted for ILSVRC was GoogleNet. 

Original Paper links - 

GoogleNet - https://arxiv.org/abs/1409.4842 <br/> 
VGG16 - https://arxiv.org/abs/1409.1556
