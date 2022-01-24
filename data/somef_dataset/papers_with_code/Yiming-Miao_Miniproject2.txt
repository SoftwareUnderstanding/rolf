# Miniproject2

# Introduction

Today's deep learning models need to be trained on large-scale surveillance data sets. This means that for each data, there will be a corresponding label. In the popular ImageNet dataset, there are one million images with manual annotations, that is, 1000 of each of the 1000 categories. Creating such a data set requires a lot of effort, and it may take a lot of people to spend months working on it. Assuming that you now have to create a data set of one million classes, you must label each frame in a total of 100 million frames of video data, which is basically impossible.

Ideally, we want to have a model that runs more like our brain. It requires only a few tags to understand many things in the real world. In the real world, I refer to classes that are object categories, action categories, environment categories, categories of objects, and many more.

# basic concepts:
The main goal of unsupervised learning research is to pre-train models that can be used for other tasks (i.e., discriminators or encoders). The characteristics of the encoder should be as generic as possible so that it can be used in classification tasks (such as training on ImageNet) and provide as good a result as possible as a supervised model.
The latest monitoring models always perform better than unsupervised pre-training models. That's because supervision allows the model to better encode features on the data set. But when the model is applied to other data sets, the supervision will decay. In this regard, unsupervised training promises to provide more general features to perform any task.
If you target real-life applications, such as unmanned, motion recognition, target detection, and real-time extraction, the algorithm needs to be trained on the video.

# Auto-encoders:
Bruno Olshausen of UC Davis and David Field of Cornell University published a paper "Sparse Coding with an Overcomplete Basis Set: A Strategy by V1?" in 1996 shows that coding theory can be used in the receiving domain of the visual cortex. They demonstrate that the basic visual vortex (V1) in our brain uses the sparsity principle to create a minimal set of basic functions that can be used to reconstruct an input image.
Yann LeCun's team is also engaged in research in this area. In the demo in the linked page, you can see how the filter like V1 learns. 
By repeating the process of greedy layer-by-layer training, a stacked self-encoder (Stacked-auto encoder) is also used.The autoencoder method is also known as the direct mapping method.

### Auto-encoders / sparse-coding / stacked auto-encoders advantages and disadvantages:

#### Advantages:<br> 
* Simple technique: reconstruct input<br> 
* Multiple layer can be stacked<br> 
* Intuitive and based on neuroscience research<br> 

#### Disadvantages:<br>  
* Each layer is trained greedily<br> 
* No global optimization<br> 
* Does not match performance of supervised learning<br> 
* Multiple layer ineffective<br> 
* Reconstruction of input may not be ideal metric for learning a general-purpose representation 

# Cluster learning
Class analysis or clustering is in the same group (called a group of objects called a task cluster in such a way), rather than those in other groups (clusters) that are more similar (in a sense) to each other. It is the main task of exploratory data mining and a common technique for statistical data analysis. It has been used in many fields, including machine learning, pattern recognition, image analysis, information retrieval, bioinformatics, data compression and computer graphics.
Cluster analysis itself is not a specific algorithm, but a general task to be solved. It can be implemented by a variety of algorithms that differ greatly in understanding what constitutes a cluster and how to find them efficiently. Popular cluster concepts include small distances between cluster members, dense areas of data space, intervals, or groups of specific statistical distributions. Therefore, clustering can be formulated as a multi-objective optimization problem. Appropriate clustering algorithms and parameter settings (including parameters such as distance functions) density thresholds or the number of expected clusters depend on the individual data set and the intended use of the results. Such cluster analysis is not an automated task, but an iterative process involving knowledge discovery or interactive multi-objective optimization of discovery and experimentation. It is often necessary to modify the data preprocessing and model parameters until the result gets the required attributes.

### Clustering learning advantages and disadvantages:

#### Advantages:<br>
* Simple technique: get the output of a similar cluster<br>
* Multi-layer stackable<br>
* Intuitive and neuroscience-based research<br>

#### Disadvantages:<br>  
* Every layer is greedily trained<br>
* No global optimization<br>
* In some cases, it can compete with the performance of supervised learning<br>
* Multi-level incremental failure == performance return diminishing<br>

# Generative models

Generative Adversarial Networks  to achieve an excellent generation model through the confrontation of the discriminator and the generator, the network hopes to be able to generate realistic images sufficient to fool the discriminator. Generating models In this area, the excellent generation of confrontation networks in recent years was proposed by Ian Goodfellow and Yoshua Bengio in the paper "Generative Adversarial Nets". 

A generated confrontation model called DCGAN, instantiated by Alec Radford, Luke Metz, and Soumith Chintala, yielded very good results. Their research is published in the paper: Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks.

The DCGAN discriminator is designed to determine whether an input picture is real (from a real picture of a data set) or false (from a generator). The generator takes a random noise vector (for example, 1024 values) as input and generates a picture.

## Generative adversarial model advantages and disadvantages

#### Advantages:<br>

* Global training for the entire network
* Easy to program and implement

#### Disadvantages:<br>  

* Difficult to train and convert problems
* In some cases, it can match the performance of supervised learning
* Need to improve usability (this is a problem for all unsupervised learning algorithms)

# Recommendation and conclusion

Unsupervised learning does not require sample data, and models can be built directly to cluster data. For unclassified things, the machine will classify the items according to certain characteristics according to their own judgment. If the robot is treated as a child, supervised learning allows him to classify the item under known rules to produce more accurate results; unsupervised learning allows the child to use the item in his or her desired way according to his or her own preference. The classification is performed to infer the internal structure of the data. When faced with unlabeled data, people can also apply unsupervised learning to allow the machine to estimate the internal structure of the item, and then label the data based on existing estimates, so that the application of supervised learning is more accurate. Classification results.

Unsupervised algorithms only deal with "features" and do not operate supervisory signals. The distinction between supervised and unsupervised algorithms is not standardized, strictly defined, because there is no objective judgment to distinguish whether the value provided by the supervisor is a feature or a goal.

In layman's terms, unsupervised learning is the majority of attempts to extract information from a distribution that does not require human annotation samples. The term is usually related to density estimation, learning to sample from distribution, learning to denoise from distribution, manifolds that require data distribution, or clustering related samples in the data.

Unsupervised learning is an important branch of machine learning. It plays an important role in machine learning, data mining, biomedical big data analysis, and data science.

In recent years, many research results have been obtained in the field of unsupervised learning, including the second winner's penalty competition learning algorithm, K-means learning algorithm, K-medoids learning algorithm, density learning algorithm, and spectral clustering algorithm; especially in gene selection. It has been widely used in the diagnosis of diseases.


# Group Members' report:<br>
#### Tingyi Zhang:<br>
Introduction of object detection model based on deep learning and CNN, including R-CNN, Fast R-CNN, Faster R-CNN, YOLO, YOLOv2, YOLOv3. R-CNN family is more based on static object detection. Faster R-CNN is the fastest among them. YOLO family is more useful in real time object detection which means that the input can be a video.

#### Xiaoyu An:<br>
Introduction of R-CNN specific four parts: Regional suggestion algorithm(ss), Feature Extraction Algorithm (AlexNet), Linear classifier (linear SVM) and Bounding box correction regression model (Bounding box). Also gives us a procedure of using R-CNN to run an object detection.

#### Vivian Pazmany:<br>
Introduction of supervised learning and unsupervised learning in machine learning. Also gives us an introduction of TensorFlow. TensorFlow offers a supervised machine learning beginners guide to classify images, train the neural network and test the accuracy of the model.

#### Yingqi Zhang:<br>
Introduction of computer vision
Introduction of objection detection
Difference between R-CNN family algorithms and YOLO family
Recommendations of using different models

#### Steven Zhu:<br>
The use of cloud computing to improve conventional deep unsupervised learning algorithms
Unsupervised learning is able to be trained without labeled data, and is able to be implemented in multiple systems once created. 
Unsupervised learning is more important when there is a lack of annotated data like medical data

### Reference
http://redwood.psych.cornell.edu/papers/olshausen_field_1997.pdf<br>
https://cs.nyu.edu/~yann/research/deep/<br>
https://cs.stanford.edu/~acoates/papers/coatesng_nntot2012.pdf<br>
https://arxiv.org/abs/1406.2661<br>
