<!-- PROJECT LOGO -->
<br />
<p align="center">
  <h1 align="center">TFLite-Frontier-Project-UPH</h1>
  
  <p align="center">
    A Light-Weight Image Classifier Android App
    <br />
  </p>
</p>

<!-- ABOUT THE PROJECT -->
## About The Project

This project is about building an Android-based image classification program using a pre-trained graph, MobileNet, which is provided by TensorFlow. This project was done with the help of "[TensorFlow for Poets 2: TFLite Android](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets-2-tflite)" and "[TensorFlow For Poets](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets)", which are tutorials from CodeLabs by Google. With this program, the user will be given a label (and the percentage of the confidence) of the object recorded real-time using the device camera. 

### Built With
* [Tensorflow's MobileNet](https://www.tensorflow.org/lite/models/image_classification/overview)
* [Google Images Download](https://google-images-download.readthedocs.io) by [hardikvasa](https://github.com/hardikvasa)
* [Autocrop](https://github.com/leblancfg/autocrop) by [leblancfg](https://github.com/leblancfg)



<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple example steps.

### Prerequisites

This program was build using these versions of framework (Older or Newer version may or may not work properly) :
* Python ver 3.7.4
* Tensorflow ver 1.14.0
```sh
pip install --upgrade "tensorflow==1.7.*"
```

### Data Scraping
We collect our data using a python script which will automatically download all images regarding the provided keyword
```sh
googleimagesdownload -k "apple" -sk "fruit" -l 1000 --chromedriver D:\chromedriver.exe
```

### Data Cleansing
Because we bulk-downloaded our data from google. Some images may be unfit to be used.
```sh
autocrop -o ./cropped/ -r ./cropped/reject/
```


### Installation

1. Clone the repo
```sh
git clone https://github.com/ZyphonGT/TFLite-Frontier-Project-UPH
```
2. Open Android Studio
3. Choose `Open Existing Project`
4. Open the project file `Project_File/android/tflite`
5. After Android Studio finishes loading, click on `Sync Project with Gradle Files`
6. Run the app

### Setting Up
Retraining The Pre-Trained Models
```sh
py -m scripts.retrain --bottleneck_dir=tf_files/bottlenecks --how_many_training_steps=500 --model_dir=tf_files/models/ --summaries_dir=tf_files/training_summaries/mobilenet_0.50_224 --output_graph=tf_files/retrained_graph.pb --output_labels=tf_files/retrained_labels.txt --architecture=mobilenet_0.50_224 --image_dir=tf_files/...
```


### CLI Usage

For development purposes, it is also possible to test the trained model using image input (.jpeg, .jpg, .gif) via Command Prompt

1. Open your Command Prompt and Navigate to the project folder
2. Enter the following command
```sh
py -m scripts.label_image --graph=tf_files/retrained_graph.pb  --image=PATH_TO_YOUR_TEST_IMAGE
```

## About MobileNet

MobileNet is a small efficient convolution neural network. "convolution" means that the same calculation are performed at each location in the image.

MobileNet is build on Depthwise Separable Convolution, which is divided into 2 kind of operation: a 3x3 depthwise convolution and a 1x1 a pointwise convolusion. the architecture is different than the "traditional" CNN's which instead using a 3x3 convolution layer.

A few things that MobileNet is more favorable beacuse they're insanely small, fast, remarkably accurate, and and easy to tune for resources vs.accuracy. which is the reason why it is so important for our project, the mobile deep learning task are mostly performed in the cloud, and this is change quickly. it is more practical to use a system that has no requirement of internet connection, it is more efficient and faster.

The standard convolutional layer is parameterized by convolution kernel K of size DK × DK × M × N where DK is the spatial dimension of the kernel assumed to be square and M is number of input channels and N is the number of output channels as defined previously.

(a) Standard Convolution Filters
![standard convolution filter](https://user-images.githubusercontent.com/35583681/61589703-4dea5800-abd8-11e9-9cef-d8c65e80a923.PNG)

![](https://miro.medium.com/max/963/1*XloAmCh5bwE4j1G7yk5THw.png)

(b) Depthwise Convolutional Filters
![Depthwise Convolutional Filters](https://user-images.githubusercontent.com/35583681/61589741-eda7e600-abd8-11e9-865c-344562dfd135.PNG)

![](https://miro.medium.com/max/963/1*yG6z6ESzsRW-9q5F_neOsg.png)

(c) 1X1 Pointwise Convolution
![1x1 pointwise  convolution filter](https://user-images.githubusercontent.com/35583681/61589781-50997d00-abd9-11e9-8ae2-dccec14f3b1c.PNG)

![](https://miro.medium.com/max/963/1*37sVdBZZ9VK50pcAklh8AQ.png)

left: standard Convolutional layer with batchnorm and ReLU. Right: Depthwise Separable convolutions with Depthwise and Pointwise layer combine with batchnorm and ReLU.
![cnn figure](https://user-images.githubusercontent.com/35583681/61589997-14b3e700-abdc-11e9-9943-d352d2a1fdf4.PNG)


## Face embeddings with MobileNet
the MobileNet neural network has been tested before by using it in FaceRecognition, as it is written inside its paper. The FaceNet model is a face recognition model, it builds the face embeddingsbased on triplet loss. using the FaceNet model, the reserahcer use distillation to train by minimizing the squared differences of the output. below is the result:
![face embbeddings](https://user-images.githubusercontent.com/35583681/61590530-fd2d2c00-abe4-11e9-87d3-1bfb49dabfc8.PNG)

Source: https://arxiv.org/pdf/1704.04861.pdf

## Team Member

Ricky - 00000020025 - rickygani10@gmail.com - Informatics 2016 Universitas Pelita Harapan
Wilbert Nathaniel - 00000019924 - wilbert.wijaya@yahoo.com - Informatics 2016 Universitas Pelita Harapan

Project Link: [https://github.com/ZyphonGT/TFLite-Frontier-Project-UPH](https://github.com/ZyphonGT/TFLite-Frontier-Project-UPH)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
* [Google CodeLab](https://codelabs.developers.google.com)
* [Google Images Download](https://google-images-download.readthedocs.io) by [hardikvasa](https://github.com/hardikvasa)
* [Autocrop](https://github.com/leblancfg/autocrop) by [leblancfg](https://github.com/leblancfg)
* [A Basic Introduction to Separable Convolutions](https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728) by [Chi-Feng Wang]
* [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision
Applications](https://arxiv.org/pdf/1704.04861.pdf) by (Andrew G. Howard Menglong Zhu Bo Chen Dmitry Kalenichenko
Weijun Wang Tobias Weyand Marco Andreetto Hartwig Adam) 

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[build-shield]: https://img.shields.io/badge/build-passing-brightgreen.svg?style=flat-square
[build-url]: #
[contributors-shield]: https://img.shields.io/badge/contributors-1-orange.svg?style=flat-square
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[license-shield]: https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square
[license-url]: https://choosealicense.com/licenses/mit
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: https://raw.githubusercontent.com/othneildrew/Best-README-Template/master/screenshot.png

