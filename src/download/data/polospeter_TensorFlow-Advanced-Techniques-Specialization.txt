# TensorFlow-Advanced-Techniques-Specialization

This repository will contain my solutions for the Tensorflow: Advanced Specialization course on Coursera.  

## Course 1: Custom Models,Layers, and Loss Functions with Tensorflow 

This first course introduces you to Tensor Flow, a popular machine learning framework. You will learn how to build a basic neural network for computer vision and use convolutions to improve your neural network.

#### Week 1: A New Programming Paradigm
Compare how the Functional API differs from the Sequential API, and see how the Functional API gives you additional flexibility in designing models. Practice using the functional API and build a Siamese network!

Good article describing the different versions of the Inception network:  
https://towardsdatascience.com/a-simple-guide-to-the-versions-of-the-inception-network-7fc52b863202

Papers about the Siamese network:  
http://yann.lecun.com/exdb/publis/pdf/chopra-05.pdf  
http://slazebni.cs.illinois.edu/spring17/lec09_similarity.pdf

Assignment: Multiple Output Models using Keras Functional API

#### Week 2: Custom Loss Functions
Loss functions help measure how well a model is doing, and are used to help a neural network learn from the training data. Learn how to build custom loss functions, including the contrastive loss function that is used in a Siamese network.


#### Week 3: Custom Layers
Custom layers give you the flexibility to implement models that use non-standard layers. Practice building off of existing standard layers to create custom layers for your models.


#### Week 4: Custom Models
Loss functions help measure how well a model is doing, and are used to help a neural network learn from the training data. Learn how to build custom loss functions, including the contrastive loss function that is used in a Siamese network.



## Course 2: Custom and Distributed training with Tensorflow

#### Week 1: Differentiation and Gradients
This week, you will get a detailed look at the fundamental building blocks of TensorFlow - tensor objects. For example, you will be able to describe the difference between eager mode and graph mode in TensorFlow, and explain why eager mode is very user friendly for you as a developer. You will also use TensorFlow tools to calculate gradients so that you don’t have to look for your old calculus textbooks next time you need to get a gradient!

Learning Objectives:

- Use various TensorFlow functions to create tensor objects  
- Describe the difference between graph-based execution and eager execution in TensorFlow  
- Use TensorFlow’s GradientTape to calculate derivatives of loss functions  

Reference: CNN for visual recognition https://cs231n.github.io/neural-networks-3/#summary

#### Week 2: Custom Training
This week, you will build custom training loops using GradientTape and TensorFlow Datasets. Being able to write your own training loops will give you more flexibility and visibility with your model training. You will also use a function to calculate the derivatives of functions so that you don’t have to look to your old calculus textbooks to calculate gradients.

Learning Objectives:

- Define the steps in a custom training loop  
- Implement custom training loops using GradientTape  
- Implement a custom training loop with data from TensorFlow Datasets  

#### Week 3: Graph mode
This week, you’ll learn about the benefits of generating code that runs in “graph mode”. You’ll take a peek at what graph code looks like, and you’ll practice generating this more efficient code automatically with TensorFlow’s tools, so that you don’t have to write the graph code yourself!

Learning Objectives:

- Describe when graph mode code is preferred over eager mode code
- Use decorators and tf.autograph to convert code into graph based code


#### Week 4: Distributed training
This week, you will harness the power of distributed training to process more data and train larger models, faster. You’ll get an overview of various distributed training strategies and then practice working with two strategies, one that trains on multiple GPU cores, and the other that trains on multiple TPU cores. Get your cape ready, because you’re going to get some superpowers this week!

Learning Objectives:  

- Explain how distributed training is different from regular model training
- Use the Mirrored Strategy to train a model on multiple GPUs on the same device
- Use the TPU Strategy to train on multiple cores of a TPU

References used in Other Distributed Strategies:
https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras


## Course 3: Advanced Computer Vision with Tensorflow

#### Week 1: Introduction to Computer Vision
Get a conceptual overview of image classification, object localization, object detection, and image segmentation. Also be able to describe multi-label classification, and distinguish between semantic segmentation and instance segmentation. In the rest of this course, you will apply TensorFlow to build object detection and image segmentation models.

Learning Objectives:

- Distinguish between object localization and object detection
- Distinguish between object detection and image segmentation
- Distinguish between semantic segmentation and instance segmentation
- Explain what is transfer learning and why it's used
- Describe design options when using transfer learning
- Implement object localization with a CNN
- Implement an image classifier with transfer learning

Papers about Image segmentation models:

Fully Convolutional Networks for Semantic Segmentation https://people.eecs.berkeley.edu/~shelhamer/data/fcn.pdf

U-Net: Convolutional Networks for Biomedical Image Segmentation https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/

DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs
http://liangchiehchen.com/projects/DeepLab.html

Mask R-CNN:
https://arxiv.org/abs/1703.06870


#### Week 2: Object detection
This week, you’ll get an overview of some popular object detection models, such as regional-CNN and ResNet-50. You’ll use object detection models that you’ll retrieve from TensorFlow Hub, download your own models and configure them for training, and also build your own models for object detection. By using transfer learning, you will train a model to detect and localize rubber duckies using just five training examples. You’ll also get to manually label your own rubber ducky images!

Learning Objectives:

- Get a conceptual overview of Regional CNN (R-CNN), Fast-RCNN, and Faster R-CNN.
- Retrieve the R-CNN model from TensorFlow hub and use it to perform object detection.
- Use TensorFlow’s object detection API to visualize the predicted bounding boxes from an object detection model
- Go beyond models in TensorFlow Hub: Load and configure a Resnet-50 model that isn’t on TensorFlow Hub, restore pre-trained weights, and select the parts of the model to retrain.
- Use the object detection API to manually annotate images for object detection
- Implement a custom training loop to fine-tune a model using just 5 training examples.

R-CNN:
https://arxiv.org/abs/1311.2524

Fast- R-CNN:
https://arxiv.org/abs/1504.08083

tENSORFLOW Hub:
https://www.tensorflow.org/hub

#### Week 3: Image Segmentation
This week is all about image segmentation using variations of the fully convolutional neural network. With these networks, you can assign class labels to each pixel, and perform much more detailed identification of objects compared to bounding boxes. You’ll build the fully convolutional neural network, U-Net, and Mask R-CNN this week to identify and detect numbers, pets, and even zombies!

Learning Objectives:

- Describe the conceptual design of fully convolutional neural networks and subsequent models based on it
- Describe the decoder section of the fully convolutional neural network
- Describe two methods of upsampling: simple scaling and transposed convolutions
- Build the encoder and decoder sections of a fully convolutional neural network
- Evaluate a segmentation model’s performance using intersection-over-union and Dice score
- Describe the conceptual design of the U-Net model
- Build a U-Net model for image segmentation
- Use the Mask R-CNN to perform instance segmentation


#### Week 4: Visualization and Interpretability
This week, you’ll learn about the importance of model interpretability, which is the understanding of how your model arrives at its decisions. You’ll also implement class activation maps, saliency maps, and gradient-weighted class activation maps to identify which parts of an image are being used by your model to make its predictions. You’ll also see an example of how visualizing a model’s intermediate layer activations can help to improve the design of a famous network, AlexNet.

Learning Objectives: 

- Explain why model interpretation is important
- Calculate class activation maps to visualize the parts of the image that a model uses to make its predictions.
- Calculate saliency maps to visualize the parts of the image that a model uses to make its predictions.
- Implement Gradient-weighted Class Activation Mapping (GradCAM) to identify parts of the image that are important in a model’s predictions.
- Describe how visualization can help to improve a model’s design
