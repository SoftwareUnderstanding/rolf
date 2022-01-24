# Distracted-Drivers-Detection

![Distracted Driver GIF](https://itstherealdyl.files.wordpress.com/2016/06/output_deb8ot.gif?w=1000)

This image detection task was hosted by State Farm two years ago on Kaggle. Though its been around, it is an excellent place to test different computer vision methods. Images of drivers are taken from the ceiling of the passenger side as they perform various actions, 10 total. Only one of these 10 states are ideal though the model created is made to predict the most probable of all 10 states per image. More info about the data and competition can be found here:
https://www.kaggle.com/c/state-farm-distracted-driver-detection.

During this project I was able to test a relatively new method of learning rate schedules published this time last year (https://arxiv.org/pdf/1608.03983). I put this through a simple VGG architechture with only one restart. Since the amount of space needed was in the tens of gigabytes (and my computer has a 16GB limit) I decided to divide the height and width of each image by 4, effectively reducing it to 1/16th the size. I was able to further shrink the amount of data needed by 1/3rd by converting the images from RGB to grayscale. Even these smaller images (160x120x1) were of lesser quality I was able to achieve 99.6% on a 20% holdout set of the original 24k photos. The model trained on this data was then used to predict almost 80k photos, the output of which was submitted to Kaggle.
