The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road

**Project flow**

The project performs behavioral cloning, training an AI agent to mimic human driving behavior in a simulator. The training data is collected by a human demonstrator driving around a track in the simulator. The vehicle's camera collects images from the human demonstration, a deep neural network is trained to predict the vehicle's steering angle. The final trained model is tested on the same test track that was run during the human demonstration but in autonomous mode. The following image illustrates the project flow:

![Image of projectflow](https://github.com/SSN15/Behavioral-Cloning--Implementaion-of-Autonomous-car-using-deep-learning/blob/master/images/projectflow.png)

**Data collection**

The training data was collected using Udacity's simulator in training mode. 
For normal driving data, I drove 2 laps around the track, in the default direction and 1 lap in the reverse direction.

For gathering recovery data, I did the following:
1. Hug the left side of the road, and drive while weaving left and right repeatedly
2. Do (1) for 2 laps
3. Do (1) and (2) while hugging the right side of the road

The images obtained from the left, right and center cameras can be seen below:

![Image of camera](https://github.com/SSN15/Behavioral-Cloning--Implementaion-of-Autonomous-car-using-deep-learning/blob/master/images/normal_cameras.png)

The collected data is then preprocessed by augmenting the data(Horizontal flipping), adding steering angle correction.
The preprocessed data is then stored in picke files.

**Deep Learning Architecture**

I implemented a deep learning model from the following reasearch paper.
http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

The visualization of the model architecture is shown below:

![Image of model](https://github.com/SSN15/Behavioral-Cloning--Implementaion-of-Autonomous-car-using-deep-learning/blob/master/images/model.png)

The model parameters are listed below:

* No of epochs= 5
* Optimizer Used- Adam
* Learning Rate- Default 0.001
* Validation Data split- 0.15
* Generator batch size= 32
* Correction factor- 0.2
* Loss Function Used- MSE(Mean Squared Error as it is efficient for regression problem)

**Result**

The model performed better with test data and the vehicle was able to complete the lap.

The final output can be seen in the video:

https://github.com/SSN15/Behavioral-Cloning--Implementaion-of-Autonomous-car-using-deep-learning/blob/master/output_video/video.mp4

**Interesting reads**

1. https://www.theverge.com/2018/5/9/17307156/google-waymo-driverless-cars-deep-learning-neural-net-interview
2. https://elitedatascience.com/overfitting-in-machine-learning
3. https://arxiv.org/abs/1812.03079
4. https://arxiv.org/abs/1512.02325
5. https://arxiv.org/abs/1711.06396
6. http://openaccess.thecvf.com/content_cvpr_2018/papers/Luo_Fast_and_Furious_CVPR_2018_paper.pdf
7. https://arxiv.org/abs/1511.00561

