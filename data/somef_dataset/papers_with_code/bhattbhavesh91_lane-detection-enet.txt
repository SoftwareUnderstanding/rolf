Lane Detection using ENet Architecture on a subset of Berkeley BDD100K Dataset
------

#### Preprocessing Steps
------
1. The dataset contains a total 500 datapoints.
2. The dataset has images & a corresponding mask showing the lane.
3. We start off with splitting the data into train, validation & test. 
4. 70% of the data is used for training purpose, while 15% is kept in validation and test.
5. Validation is performed to check if the data split has happened correctly or not by superimposing the mask on the actual image and viewing the image.

#### Creating Generator & Image Augumentation functions
------
In orderto prevent loading all the images onto the disk at once a generator is created. The function would read a batch of images and their corresponding masks from the disk and apply the following transformations:
- Converting the mask into a binary image having channels equal to the number of classes which in our case is 2 (lane area is class 1 or pixel value 255 & no-lane is class 0 or pixel value 0) 
- Resizing the image and teh maks to 720 x 1204.

Additionally, We have just 500 datapoints & there are chances we may overfit the training data. In order to avoid overfitting & create new samples we use relevent image augumentation techniques for training generator to increase our dataset size. Not all image augumentation techniques maybe suitable in case of lane detection such as vertical flipping of an image.
- Some of the image augumentation techniques used are change brightness, horizonatal flig & zoom to a particular resolution.

#### ENet - A Deep Neural Architecture for Real-Time Semantic Segmentation
------
The Research paper for Enet can be found at the link: https://arxiv.org/abs/1606.02147  
**ENet** was chosen as the algorithm for the purpose becuase it claims to provide the same accuracy as well known models such as FCN and UNet while having 2 orders less of trainable parameters.

#### Below attached is a sample output trained for 10 epochs
![Model Prediction](https://github.com/bhattbhavesh91/lane-detection-enet/blob/master/images/sample_output.jpg)
