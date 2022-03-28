# Cartoonizer Android App

This is an Android  app with the White-box CartoonGAN TensorFlow Lite models , cycleGAN & styleGAN2.   

## TensorFlow Lite Model
There are three TensorFlow Lite Models included in the Android app and see the [ml](../ml/) README for details.  
Android Studio ML Model Binding was used to import these models into the Android project.

## drive link for apk and demo
drive: https://drive.google.com/drive/folders/1jNj-ao5Ybb5HxuKZ3ZFKmmn04Sx2aOsv?usp=sharing

## Requirements
* Android Studio Preview Beta version - download [here](https://developer.android.com/studio/preview).
* Android device (with at least 3GB RAM) in developer mode with USB debugging enabled
* USB cable to connect an Android device to computer

## Build and run
* Clone the project repo:  
`git clone https://github.com/margaretmz/CartoonGAN-e2e-tflite-tutorial.git`  
* Open the  code in Android Studio.
* Connect your Android device to computer then click on `"Run -> Run 'app'`.
* Once the app is launched on device, grant camera permission.
* sign up to launch the app.
* select the type you want (cartoonGAN , styleGAN2 , style transfer, filters).
* Take a selfie or a photo and wait to process. 


## features
1- cartoonGAN
2- StyleGAN2
3- CycleGAN
4- Style transfer
5- Filters
6- Augmanted reality


## cartoonGAN
The white-box CartooGAN TensorFlow Lite models (with metatdata) are available on TensorFlow Hub in three different formats:
Dynamic-range
Integer
float16

![image](https://user-images.githubusercontent.com/60838458/126578239-7c5c7afb-6044-4312-b5a0-192b53e6dc75.png)


paper: https://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_CartoonGAN_Generative_Adversarial_CVPR_2018_paper.pdf

helpful resources: https://blog.tensorflow.org/2020/09/how-to-create-cartoonizer-with-tf-lite.html

## StyleGAN2

the classic StyleGAN model which is trained on photos of people’s faces. This was released with the StyleGAN2 code and paper and produces pretty fantastically high-quality results.
the faces model were fine-tuned on a dataset of various characters from animated films. It’s only around 300 images but enough for the model to start learning what features these characters typically have.
there are two API for the feature 

RapidAPI: https://rapidapi.com/toonify-toonify-default/api/toonify

DeepAI: https://deepai.org/machine-learning-model/toonify

colab: https://colab.research.google.com/drive/1s2XPNMwf6HDhrJ1FMwlW1jl-eQ2-_tlk?usp=sharing

paper: https://paperswithcode.com/method/stylegan2

![image](https://user-images.githubusercontent.com/60838458/126884084-22b51924-3f07-4a01-a76c-dcad8e6fd4fb.png)


## CycleGAN
The model was built same as the model architecture described in the official cycleGAN paper and used across a range of image-to-image translation tasks.
The implementation used the Keras deep learning framework based directly on the model described in the paper and implemented in the author’s codebase, designed to take and generate color images with the size 256×256 pixels. The architecture is comprised of four models, two discriminator models, and two generator models.

my colab training: https://colab.research.google.com/drive/10ZOGAcqytp2wm-e9wk7vGvwAwBsMcsyv#scrollTo=bKgkrXgvc5If

drive link for tflite model: https://drive.google.com/drive/folders/1fr60j9GaVp0j3Ccl40X0fHDQb45ytCdD?usp=sharing

refrence that i followed: https://machinelearningmastery.com/cyclegan-tutorial-with-keras/

paper: https://arxiv.org/abs/1703.10593

CycleGAN github: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

book reference that hepls me alot to under stand cycleGAN: GANs in action

book link: https://www.amazon.com/GANs-Action-learning-Generative-Adversarial/dp/1617295566

## Style transfer
style transfer model from TensorFlow lite was used in the project as a trained model for the application.

tensorFlow lite for style transfer: https://www.tensorflow.org/tutorials/generative/style_transfer

model link: https://www.tensorflow.org/lite/examples/style_transfer/overview

![image](https://user-images.githubusercontent.com/60838458/126884294-be707361-d415-4bc1-a1e6-065e1ee258ab.png)


output:

![image](https://user-images.githubusercontent.com/60838458/126884297-f97f4baa-e47f-4f56-9170-0dc294fa4ad5.png)


## Filters
filters library: https://github.com/nekocode/CameraFilter

![image](https://user-images.githubusercontent.com/60838458/126884326-a1d4eee6-5dc7-425a-befb-7de0a6d6126f.png)


## things that helped me in general
1- courses:

Neural Networks and Deep Learning: https://www.coursera.org/learn/neural-networks-deep-learning

Convolutional Neural Networks: https://www.coursera.org/learn/convolutional-neural-networks

Neural Style Transfer with TensorFlow: https://www.coursera.org/projects/neural-style-transfer

2- books:
deep learning principles.
GANs in Action.

## note
if you are egyptian or someone who can't open medium website you can use this extension:
https://chrome.google.com/webstore/detail/browsec-vpn-free-vpn-for/omghfjlpggmjjaagoclmmobgdodcjboh?hl=en
