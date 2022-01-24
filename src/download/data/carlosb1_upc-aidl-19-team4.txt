# Introduction

*student: Carlos Báez*


Project report for the Deep learning postgrade at UPC tech talent (Barcelona). This report explains all the work done, results and extracted conclusions

The main idea in the project was the implementation of an end-to-end person recognition system. For this, I decided to split the project in two parts:


- **Detection**. Study of different implemented algorithms and different datasets to choose the best option for us
   
- **Face Recognition**. It is implemented and modified four different solutions with a saimese architecture.

## Motivation

At the beginning, my main motivation was the implementation of a complete pipeline for people recognition, where I analysed the different parts: *detection* and *recognition*. In the moment to work with recognition I liked the  Siamese networks[1] and how they improve the performance then  I decided to review it.

After this, I started to be interested in how a retrieval system can work and can be scalable applying cosine functions[2]. With this code, I could figure out how the extraction of features has a powerful role int this type of solution.


# Structure Code
```
pipeline                                      -> Main source folder
├── detections                                -> Detection pipeline
│   ├── db                                    -> Datasets classes to configure dataset 
│   │   ├── constants.py                      -> Necessary constants for dataset classes
│   │   ├── FDDB_convert_ellipse_to_rect.py   -> Parser from ellipse to rectangle for FDDB dataset
│   │   ├── FDDB_dataset.py                   -> FDDB dataset class which loads in memory all the dataset information
│   │   ├── FDDB_show_images.py               -> Example to display FDDB dataset
│   │   └── WIDERFaceDataset.py               -> Wider dataset class example 
│   ├── Tiny_Faces_in_Tensorflow              -> Folder for Tiny Faces model
│   │   ├── tiny_face_eval.py                 -> Entrypoint for tiny_faces model (Tensorflow)
│   └── yolo                                  -> Folder for YOLO model 
│       └── yolo                              -> Folder for YOLO package
│           ├── model.py                      -> YOLO model
│           └── yolo.py                       -> YOLO class, functions to call inference and high level functions for detection
├── README.md                                 -> Main README.md
├── recognition                               -> Recognition pipeline
│   ├── cfp_dataset.py                        -> CFP Dataset class
│   ├── metrics.py                            -> Class to calculate threshold and accuracy
│   ├── metrics_retrieval.py                  -> Class to implement ranking
│   ├── models.py                             -> Class with different models
│   ├── params.py                             -> Builder params pattern to customize different tests
│   ├── parse_cfg_dataset.sh                  -> Script to fix dataset paths
│   ├── tests.py                              -> Class to execute different tests
│   ├── train.py                              -> Main class which train loop
│   ├── transforms.py                         -> Data augmentation classes
│   └── utils.py                              -> Functions for different use cases
└── scripts                                   -> General scripts
    ├── evaluate_tiny_faces.py                -> Script to execute and evaluate tiny faces
    ├── evaluate_yolo.py                      -> Script to execute and evaluate yolo
    ├── get_models.sh                         -> Download YOLO weights
    ├── README.md                             -> README with dependencies
    └── utils.py                              -> Other functions
scripts                                       -> General scripts
├── graphs_model.ipynb                        -> Draw seaborn bubble graph
├── local2remote.sh                           -> Script to upload from local to one server
├── print_graphs.ipynb                        -> Draw matplot graphs
├── remote2local.sh                           -> Script to download from remote to local 
├── test_practica_carlos.ipynb                -> DEMO. First version of the final demo. 
├── train.ipynb                               -> Collab to set up environment and ssh connection
└── value_models.csv                          -> Information to print in graphs ()

```


# Documentation

## Detection
For the detection module. It was studied and analysed two neural networks and two datasets:
- *Tiny Faces*
	- code https://github.com/cydonia999/Tiny_Faces_in_Tensorflow
	- paper https://arxiv.org/abs/1612.04402
- *YOLO v3*, trained for faces
	- code https://github.com/sthanhng/yoloface
	- paper https://pjreddie.com/media/files/papers/YOLOv3.pdf


For datasets, I did two differents:

- *FDDB* Dataset http://vis-www.cs.umass.edu/fddb/
- *Wider* Dataset http://shuoyang1213.me/WIDERFACE/ 


The bubble graph can give us a small overview about the differences of both: (accuracy, time and number of parameters for each network):

![alt text][bubbles]

## Recognition

It was implemented a Siamese Network with [VGG][vgg] features. I got a pretrained VGG with Imagenet[3] and I applied a finetuning for faces.

In general, I implemented  different networks with different loss techniques:
- Two siamese neural networks getting features from a VGG convolutional network and the application of a cosine similarity[5]
- Two siamese networks which a concatenation in order to join features and get a classification with  a cross entropy loss[4]
- One siamese with a triplet loss function

About experiments, they are classified as:
- Change optimizer SGD or ADAM (With different learning rates and weight decay) (1e-3, 5e-4, 1e-4)
	- It was tuned other parameters as weight decay, betas, momentum, etc... In order to find the best configuration that I added in the result table
- With and without data augmentation. In the data augmentation process with rotations, flips and jitter modifications.
	- The idea is check if they have improvements. If it happens, add more modifications to improve the percentage.
- Changing the loss functions that means change the type of neural network

## Recognition architecture
The backend architecute is a VGG16-bn (batch normalized) and its convolutional layers. They are used as a siamese network applying them in two images and get their features. For this project, it is used pretrained networks that speed up our training process with a pretrained neural network with Imagenet

After this point, it is applied different techniques to check the performance and compare results:
- First one, it applies a cosine similarity loss function to search better results with the convolutional layers
	- v1 It is the simplest version, it only gets the VGG feature and It is applied the cosine loss function.
	- v2 In this version, it is added a linear layer to flat the features that it is trained. Furthermore, It uses the cosine  loss function too.
- In the second one, it is joined the two branches to get a classification. Furthermore, It is added improvements in order to achieve a better solution.
	- The neural network  named decision, it includes a minimal decision network with a few linear layers to do it. It is done  after the concatenation of features (from the two branches)
	- In the decision network linear, it is added a linear layer before this concatenation to improve the training and the performance. It tries to get better feature for our use case.



### VGG backend

In this image, I can preview the VGG architecture and its convolutional module. It can give us an idea where It is extracted my features for the neural networks.
![alt_text][vgg_arch]

[VGG backend architecture][vgg_features]

### Siamese Cosine immplementation

Previous architectures are depicted in the following schematics.

Two siamese cosines are very similar but the second one doesn't reuse VGG weights.. It gets worse the performance.

#### Siamese Cosine 1
![alt_text][siamese1_layers]
#### Siamese Cosine 2
![alt_text][siamese2_layers]


In the second type of architectures, they include the concatenation and the decision network to classify. The second done is adding an extra linear layer to train.
#### Decision
![alt_text][decision_layers]
#### Decision linear
![alt_text][decision_linear_layers]


### Result table

In order to evaluate which algorithm can fit better, I did different tests:
- The chosen dataset is the [cfp dataset][cfp_dataset]. It includes annotations for different or same pair of faces.
- The result table has the validation accuracy for the dataset, the idea is the calculation of the test accuracy (usin the splitted test dataset) for the best option of all.
- The table includes results of the tests but It was done some experiments to figure out how to tune parameters as the learning rate.
- The data augmentation applies jitter, flip and rotations for our images. 
- The table includes the best accuracy with the best input hyperparameters that I could find.

Here, it is the table of results for the validation split:


|      Name           | SGD    | SGD + Data aug | Adam + Data aug |
|---------------------|--------|----------------|-----------------|
| Cosine v1           | 81.14  |      80.53     |    73.03        |
| Cosine v2           | 71.35  |      73.03     |    70.75        |
| Decision            | 79.35  |      80.6      |    49.80        |
| Decision linear     | 78.28  |      81.71     |    76.75        |
| Cosine v1 + triplet |        |      83.28     |    81.71        |


#### Triplet loss results (Best results)

The winner in the benchmark is the **Cosine v1 + Triplet +  SGD optimizer and Data augmentation**. With this choosen neural network, it is tested with the test data set where it is got these results:

|      Name                        |  Validation accuracy | Test accuracy |
|----------------------------------|----------------------|---------------|
| Cosine v1 + triplet + SGD + DA   |   83.28              |    **86.32**  | 

### Siamese cosine tests (V1 and V2)


#### Cosine networks SGD test

First experiments that I did is applying SGD to obtain first results that I will be able to compare with different configurations. Here, It is possible to check how it learns without problems. 

![siamese1_sara_sgd]
![siamese2_sara_sgd]

#### Cosine networks SGD + Data aug. test

The data augmentation helps in a better training. It is possible to check how the validation and training data are fitting better.

![siamese1_sara_sgd_normtrans]
![siamese2_sara_sgd_normtrans]

#### Cosine networks ADAM + Data aug. test

Furthermore, the Adam optimizer works well with cosine networks. It is possible to check how it is improved the process to find the best loss. Unfortunately, The accuracy was poor, I tried different values for the learning rate, weight decay (0, 0.001, 5e-4) but It doesn't help, I got the conclusion I need more time to find the best hyperparameters for our case. For this, I stopped this study line.

![siamese1_sara_adam_normtrans]
![siamese2_sara_adam_normtrans]

### Cosine v1 with triplet loss! (SGD and ADAM) + Data augmentation

My last test was the implementation of the triplet loss where I got the best results. The idea to use negative and positive images in the loss function provide more comparative information to the loss function (para metric was used by default, in this case 1.0)

![triplet1_sara_sgd_normtrans]
![triplet1_sara_adam_normtrans]

### Decision networks (Decision and linear network)

I did the same experiments for the decision layers. In the first experiments, I could already detected how the performance is poor and after more experiments I could confirm it.

#### Decision networks SGD 

It is possible check how the overfitting happens very fast, and I starts to figure out that It is not the best workflow in my use case.

![decision_sara_sgd]
![decision_linear_sara_sgd]


#### Decision networks SGD + Data aug

Here, I figured out that the data augmentation is not improving the values, the overfitting only happens some epochs after. 

![decision_sara_sgd_normtrans]
![decision_linear_sara_sgd_normtrans]

#### Decision networks Adam + Data aug

Applying Adam, in this case, was exhausting... I tried different hypeparameters values but the accuracy was not better.

![decision_sara_adam_normtrans]
![decision_linear_sara_adam_normtrans]

## Conclusions

- In general, Siamese cosine v1 works better. 
- The Cosine similarity loss works better than any type cross entropy.
- A Backend pretrained architecture seems a good workflow to research more about this
- **The best recipe: cosine v1 + SGD + Data augmentation + Triplet loss** [weights]
- Decision layers have problems to train with the dataset, the overfit appears very fast (7 or 6 epoch). It is very  important to tune params and add data augmentation.
- In this particular case, I had problems to find the best parameters. In other cases, It can works well (as siamese cosine networks), but It seems that It is necessary more time for a good tuned.
- In general, decision networks need more epochs to learn better due to train the decision network and new layers.

## Use code

Installation (for python 3.6+)
```
pip install -r requirements.txt

```

The code has different entrypoints for different use cases (detection, recognition, creation of graphs, parsing, upload data). The main split of the work  is in two main
use cases: *detection* and *recognition* where they are:
	
- Detection - `evaluate_tiny_faces.py` and `evaluate_yolo.py` to execute the benchmark and run the detection algorithms
- Recognition - `test.py` is the script to train a new neural network. For the triplet, it is the last one implemented, it needed a set of important changes in the architecture, for this reason, it is created a different file `test_triplet.py`

It is important to comment that I didn't add argument line parser because It was not clear the requirements while I was developing.. For this reason, you must change different
paths (datasets, weights, etc..) paths for your environment.

Then, to execute the training It must be something like this (for python 3.6+):
```
# In the recognition directory
python tests.py
```
for triplet training similar:
```
# In the recognition directory
python test_triplet.py
```

- If you must change parameters, you change the Builder Params pattern, it is used to customize your parameters[3]

**NOTE**:  It is obvious that the code has technical debt, my main effort was to find the best architecture and parameters.. The code needs to be refactorized.

To get the validation and test accuracy for recognition. From the recognition folder, you can execute `metrics.py` (for python 3.6+)

```
python metrics.py path_saved_model_file [threshold]
``` 

If you add the threshold, it will calculate the accuracy taking care the argument, otherwise it will calculate the best threshold for the dataset and  calculate both accuracies.

The demo is included in `scripts/test_practica_carlos.py`

**WEIGHTS:**
- YOLO WEIGHTS: https://drive.google.com/open?id=1ZsWJx2IwMTO7WZrlyC6Bg4G-yUA5Vhd1
- TINY_FACES WEIGHTS: https://drive.google.com/open?id=18NuCfWNScDpCr9Un3yuhuE6BNpf_2-0e

## References
[1] Siamese networks https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf

[2] Cosine loss https://pytorch.org/docs/stable/nn.html

[3] Imagenet http://www.image-net.org/

[4] Cross entropy loss https://pytorch.org/docs/stable/nn.html

[5] Cosine similarity https://pytorch.org/docs/stable/nn.html

[Facenet] https://github.com/davidsandberg/facenet

[OpenFace] https://cmusatyalab.github.io/openface/

[YoloV3_paper] https://pjreddie.com/media/files/papers/YOLOv3.pdf

[TinyFaces_paper] https://arxiv.org/abs/1612.04402

[VGG_paper] https://arxiv.org/abs/1409.1556

[data_augmentation] https://github.com/aleju/imgaug

[triplet_loss]https://en.wikipedia.org/wiki/Triplet_loss



[bubbles]: https://github.com/carlosb1/upc-aidl-19-team4/blob/master/resources/bubbles.png "Bubbles"
[cfp_dataset]: http://www.cfpw.io/ "CFP Dataset"
[weights]: https://drive.google.com/open?id=1s3Zj0PesMp2juGmS7ERd5GWvxuxk-u2D
[vgg]: https://arxiv.org/pdf/1409.1556.pdf
[decision_layers]: https://github.com/carlosb1/upc-aidl-19-team4/blob/master/resources/decision_layers.png "Decision layers"
[decision_linear_layers]: https://github.com/carlosb1/upc-aidl-19-team4/blob/master/resources/decision_linear_layers.png "Decision linear layers"
[decision_linear_sara_adam_normtrans]: https://github.com/carlosb1/upc-aidl-19-team4/blob/master/resources/decision_linear_sara_adam_normtrans.png "Decision linear Adam  + Data Augmentation"
[decision_linear_sara_sgd]: https://github.com/carlosb1/upc-aidl-19-team4/blob/master/resources/decision_linear_sara_sgd.png "Decision linear SGD"
[decision_linear_sara_sgd_normtrans]: https://github.com/carlosb1/upc-aidl-19-team4/blob/master/resources/decision_linear_sara_sgd_normtrans.png "Decision linear  SGD + Data augmentation"
[decision_sara_sgd]: https://github.com/carlosb1/upc-aidl-19-team4/blob/master/resources/decision_sara_sgd.png "Decision SGD"
[decision_sara_adam_normtrans]: https://github.com/carlosb1/upc-aidl-19-team4/blob/master/resources/decision_sara_adam_normtrans_lr54.png "Decision Adam  + Data Augmentation"
[decision_sara_sgd_normtrans]: https://github.com/carlosb1/upc-aidl-19-team4/blob/master/resources/decision_sara_sgd_normtrans_v2.png "Decision SGD  + Data Augmentation"
[siamese1_layers]: https://github.com/carlosb1/upc-aidl-19-team4/blob/master/resources/siamese1_layers.png "Siamese Cosine 1 layers"
[siamese1_sara_sgd]: https://github.com/carlosb1/upc-aidl-19-team4/blob/master/resources/siamese1_sara_sgd.png "Siamese Cosine 1 SGD"
[siamese1_sara_adam_normtrans]: https://github.com/carlosb1/upc-aidl-19-team4/blob/master/resources/siamese1_sara_adam_normtrans_lr54.png "Siamese Cosine 1 Adam  + Data Augmentation"
[siamese1_sara_sgd_normtrans]: https://github.com/carlosb1/upc-aidl-19-team4/blob/master/resources/siamese1_sara_sgd_normtrans.png  "Siamese Cosine 1 SGD  + Data Augmentation"
[siamese2_layers]: https://github.com/carlosb1/upc-aidl-19-team4/blob/master/resources/siamese2_layers.png  "Siamese 2 layers"
[siamese2_sara_adam_normtrans]: https://github.com/carlosb1/upc-aidl-19-team4/blob/master/resources/siamese2_sara_adam_normtrans.png "Siamese Cosine 2 Adam  + Data Augmentation"
[siamese2_sara_sgd]: https://github.com/carlosb1/upc-aidl-19-team4/blob/master/resources/siamese2_sara_sgd.png  "Siamese Cosine 2 SGD"
[siamese2_sara_sgd_normtrans]: https://github.com/carlosb1/upc-aidl-19-team4/blob/master/resources/siamese2_sara_sgd_normtrans.png  "Siamese Cosine 2  SGD  + Data Augmentation"
[vgg_arch]: https://github.com/carlosb1/upc-aidl-19-team4/blob/master/resources/vgg_arch.png "VGG architecture"
[vgg_features]: https://github.com/carlosb1/upc-aidl-19-team4/blob/master/resources/vgg_features.png "VGG features"

[triplet1_sara_adam_normtrans]: https://github.com/carlosb1/upc-aidl-19-team4/blob/master/resources/siamese1_sara_triplet_adam_normtrans.png "Siamese Triplet 1 Adam  + Data Augmentation"
[triplet1_sara_sgd_normtrans]: https://github.com/carlosb1/upc-aidl-19-team4/blob/master/resources/siamese1_sara_triplet_sgd_normtrans.png  "Siamese Triplet 1 SGD  + Data Augmentation"



