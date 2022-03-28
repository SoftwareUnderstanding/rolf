## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

## 1. Introduction
The aim of this write up is to describe the process of training a convolutional neural network to complete at least one lap of bot tracks provided. I will begin by setting the goals of the project, then I will give a walkthrough of the process, the decisions I made and parameters chosen. I will then describe the results of the process, followed by a discussion about the limitations and possible improvements.

![Original image](./images/orig_image.jpg)

## 2. Goals
The goals / steps of this project are the following:
* Data gathering
* Data preprocessing
* Data generation
* Model architecture
* Train the model
* Results

## 3. Walkthrough

#### 3.1 Data gathering
The data gathering step is one of the most important parts of the process, poor data will confuse the network resulting in longer convergence times, less stability in the final model and erratic driving behavior. When gathering data I concentrated on providing smooth steering input so the model would have reliable training data. I made an effort to avoid over steering as this requires counter steering to correct the vehicles position which would generate contradictory data. A number of track locations proved problematic for the model primarily the the right hand turn (track 1), the start of (track 2), steep downhill sections (track 2) and shadowed turns (track 2). I dealt with these locations by recording extra data throughout these sections, and where required recording recovery data, where I placed the car in a position as if it was leaving the road and recorded the steering required to recover the situation. Another issue with the data gathering process is that track 1 is driven counter clockwise resulting in a majority of left hand turns, this can cause the model to become bias towards the more common data points. My solution was to drive both tracks forward and backwards resulting in a more balanced dataset.  

![Problematic locations](./images/problems.jpg)

#### 3.2 Data preprocessing
The images that the model is being trained on now need to be cleaned so that the information we are interested in can be more easily identified. We achieve this by cropping the images so that the majority of irrelevant features are removed, for example we crop the lower pixels of the image because they mainly consists of the front of the vehicle and we remove the top portion of the image because it largely consist of sky and features that are not directly related to the road. Next we blur the images to reduce noise that can become a problem during training when a model can overfit the training data, essentially it starts to identify irrelevant noise as features. Finally we resize the images so that they are the correct input size for the network. This process must also be performed on any new data passed to the model.

![Preprocessing steps](./images/process.jpg)

#### 3.3 Model Architecture
The model architecture I deemed to be most appropriate was the network developed by nVidia described in the article 'End-to-end deep learning for self driving cars'. This network consists of a normalization and mean centering layer followed by five convolutional layers and finally by three densely connected layers, I add a dropout layer between the convolutional layers and the dense layers to reduce overfitting and make the model more robust. After some research I choose exponential linear unit (ELU) for the activation function because it "speeds up learning in deep neural networks and leads to higher classification accuracies" (Clevert, Unterthiner, Hochreiter, Cornell University 2016). Beyond adding the dropout layer I did not tune the parameters in the network as it has been heavily tuned by nVidia to predict on steering values given images.

![Network architecture](./images/cnn.png)

#### 3.4 Data generation
The gathered data points consist of a steering angel and 3 images, the images come from front facing cameras positioned on the left, center and right of the vehicle. The steering angle is adjusted by -0.2 for the right image and +0.2 for the left image thereby generating three time the data per run. Intuitively the next step would be to load all our gathered data and begin training our model but there is a more effective approach. Neural networks tend to perform better on very large datasets, it is not feasible to collect vast amounts of data manually so I implemented a generator which is a means of generating an arbitrary amount of data from a limited pool. To begin we load the gathered data into RAM, approximately 21,000 data points), the generator then chooses a random image and adjusts it within given parameters creating an entirely new (but still valid) image, the steering measurement is also adjusted accordingly, in this way we can vastly increase the size of the dataset. For the final model I choose to multiply the dataset by a factor of 5 as I observed diminishing improvements beyond this.

The process of altering the images was to shift all the pixels in the image left or right by a number not greater than 25 and to adjust the steering angle by 0.002 to remain consistent with the altered image (both of these parameters were tuned by trial and error). Next there is a fifty percent chance the image is flipped horizontally and the steering measurement is multiplied by negative one.
This method of generating data on the fly is also useful due to limited memory on many systems, the data can be continuously generated in relatively small batches and only the batch size multiplied by a data point size needs to be stored in RAM.

![Image alteration steps](./images/shift.jpg)

#### 3.5 Train the model

The hyperparameter were either fine tuned through trial and error or set at commonly used values for example the training/validation split is set at 80/20 as anything from 80/20-70/30 is standard practice. Whereas the training set was entirely generated data (altered images and measurements) the validation set was sampled from non-adjusted images to ensure the most accurate prediction metric possible. The batch size of 64 is large enough that the model is not backpropagating to often causing training times to increase but small enough that the model is adequately tuned this was determined by trial and error. I observed significant improvement when I increased the training data via the multiplier up to 4-5 (up to approximately 100,000 data points) but beyond this I saw diminishing improvements. The number of epochs is set to 5 because on previous iterations of the model I found that the model begins to overfit the training data, this is illustrated in the graphs generated. I chose 'Adam' as the optimizer as it has adaptive learning rates methods to find individual learning rates for each parameter it is essentially a combination of RMSprop and Stochastic Gradient Descent with momentum (paraphrased, Vitaly Bushaev, 2018).

![Train/Validation loss per epoch](./images/model_track2_5mul_5eph.png)

## 4. Results/Discussion
Using the model trained on data from tracks 1 and 2 driven both forward and backwards as well as four sets of recovery data the car completes both track 1 and 2 without incident. The car swerves left and right slightly on the straights this is probably due to the close proximity of the three cameras mounted on the car, a reduction in the 0.2 correction value might mitigate this effect. I believe that more training data would make this model more robust. The nVidia team used YUV colour space in their network but I observed no discernible improvement over RGB. There are a number of preprocessing steps that could help improve the model the for example incorporating canny edge detection should stand to better identify the road edges. Adding random lightness variation to the generated training data may help to improve the model when dealing with shadows and low light levels.


## 6.0 References:
https://devblogs.nvidia.com/deep-learning-self-driving-cars/
https://www.geeksforgeeks.org/image-processing-in-python-scaling-rotating-shifting-and-edge-detection/
https://www.ingentaconnect.com/contentone/ist/ei/2017/00002017/00000019/art00009?crawler=true&mimetype=application/pdf
https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
https://sefiks.com/2018/01/02/elu-as-a-neural-networks-activation-function/
https://arxiv.org/abs/1511.07289
https://towardsdatascience.com/adam-latest-trends-in-deep-learning-optimization-6be9a291375c
https://medium.com/@karpathy/a-peek-at-trends-in-machine-learning-ab8a1085a106
