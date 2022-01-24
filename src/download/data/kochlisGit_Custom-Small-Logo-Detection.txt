# Custom Small Logo Detection
Logo Detection of a custom small dataset. In this project, I created a custom dataset with images from Google and then I trained a model, in order to detect any of the known logos in images and video streams. I used a state of the art model, which is both fast and performs high accuracy.

# Dataset
The dataset contains 108 Images with logos of 6 famous brands (**classes**): Nike | Jordans | Adidas | Puma | Kappa | Quicksilver. 

![](https://github.com/kochlisGit/Custom-Small-Logo-Detection/blob/main/JPEGImages/adi12.jpg)
![](https://github.com/kochlisGit/Custom-Small-Logo-Detection/blob/main/JPEGImages/jordans11.jpg)
![](https://github.com/kochlisGit/Custom-Small-Logo-Detection/blob/main/JPEGImages/kappa13.jpg)
![](https://github.com/kochlisGit/Custom-Small-Logo-Detection/blob/main/JPEGImages/nike11.jpg)
![](https://github.com/kochlisGit/Custom-Small-Logo-Detection/blob/main/JPEGImages/puma5.jpg)
![](https://github.com/kochlisGit/Custom-Small-Logo-Detection/blob/main/JPEGImages/quicksilver8.jpg)

There are 5 big challenges in this particular dataset.

1. **The LOGOs could be displayed in any size in a frame.** As You can notice, some logos have bigger size than others and every logo has different size in every image.
2. **The LOGOs could be rotated and even located everywhere in the image.** In the MNIST Digit dataset, all digits are placed in the center of the image. However, in this case, logos could be everywhere, without having fixed size.
3. **The color of the logos could be different.** For example, Nike's logo could be white/black/yellow, etc.
4. **The dataset is very small,** because It was made by manually downloading from Google. Usually, when a client requests a LOGO Detection model, they don't provide a lot of images, which are necessary for the training of the model.
5. **The model needs to be fast enough to detect LOGOs in a video stream.** However, the faster a network becomes, the lower it performs. There are state of the art models with very high performance in low fps, such as **Hourglass CenterNet**, however, this model has tons of parameters (1.3 GB) and requires a lot of time and a lot of money for good hardware.

# Technologies
For this particular project, I used the **Object Detection API by Tensorflow**. This package provides state-of-the-art models, that can be trained with little effort. Also, their implementation of the models is highly optimized, by Tensorflow team.

# Model
I used the EfficientDet-v4: https://arxiv.org/abs/1911.09070 . According to Tensorflow, It can run in 133 fps in a fast computer and it performed 48.5 mAP (Mean Average Precision) in COCO dataset. This is a very high accuracy for a model. Also, If It can perform in 133 fps, then It means that It could process about 8 frames per second, which is great for a video stream. 

# Configurations
- Changed number of classes from 90 to 6, because our dataset contains only 6 classes.
- Replaced random initialization of weights with Xavier Initialization. This type of initializer has shown that it can prevent exploding weights.
- Used the already pretrained EfficientDet model. This model was trained in the COCO dataset, in which it has seen 1.5 million images. However, since this is a detection problem, I set fine tune type to **DETECTION**
- Reduced the batch size from 128 to 16. It is important to keep the batch size low. Models that are trained with low batch sizes have shown to train faster and perform better than those with big batch sizes. However, result in faster training.
- Set the number of epochs to 70000. This should be enough for our small dataset.
- Replaced the momentum optimizer with ADAM. Adam optimizer is an improvement of momentum optimizer. It combines: SGD, ADAGrad, RMSProp. 
- Added Data Augmentation techniques: Random Horizontal Flip | Random Crop | Random Saturation | Random Brightness | Random Hue | Random Contrast. This is how I dealt with most of the challenges of the dataset.
- Added shuffling in the training inputs. In each epoch, the model will shuffle the inputs. This is a quick way of dealing with overfitting. An alternative could be Cross Validation.

Finally, It took 12 hours for the model to be trained with a TPU in the Google Cloud. 

# Results
![](https://github.com/kochlisGit/Custom-Small-Logo-Detection/blob/main/Validation-Results/adidas.png)
![](https://github.com/kochlisGit/Custom-Small-Logo-Detection/blob/main/Validation-Results/jordans.png)
![](https://github.com/kochlisGit/Custom-Small-Logo-Detection/blob/main/Validation-Results/puma.png)
![](https://github.com/kochlisGit/Custom-Small-Logo-Detection/blob/main/Validation-Results/kappa.png)

Notice how easily It manages to detect any logo, no matter where it placed, or how big it is, or how it is rotated in the image. I also created a script that uses the webcam to detect logos, so that I could try it by myself. However, the video is a bit laggy, because I run it on GTX 980, which is an old GPU with not much Computational power.

# 4K-Walk in Spain
https://drive.google.com/file/d/1q6wynkG7FTpHaGh_hmi4zgJ-rlqBR_wG/view?usp=sharing

# Validation from Webcam
https://drive.google.com/file/d/1dQEsU6b4v1a2D6r1F8-AuFk6tKOPY07O/view?usp=sharing

# Graph
Below, I will provide you with the model's graph, in case You are interested in downloading the model:

https://drive.google.com/drive/u/0/folders/1LGaoijE-VuFYufbCIkC6oFrgswfkIiIw
