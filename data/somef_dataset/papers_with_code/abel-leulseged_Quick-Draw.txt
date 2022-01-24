# Quick-Draw
## By Abel Tadesse, Richy Chen, Jason Yi

## Brief Description:
Quick, Draw! game was originally an experimental game released by Google to teach the general population about artificial intelligence. The game asks the user to draw a picture of an object or idea and uses a neural network artificial intelligence to guess what the sketch represents. Moreover, the network continues to guess as the user draws, even though the sketch is incomplete. The network uses such drawings to learn and increase its accuracy in the future.

## Kaggle Competition/The Problem:
To begin, the training data comes from the actual Google game itself. Consequently, they may be incomplete or may not even represent the label. As a result, the task is to build a better classifier using the noisy dataset. 

## Data:
We are given two datasets from Kaggle. The raw data (65.87 GB) is the exact input recorded from user drawing. The simplified data (7.37 GB) removes unnecessary information from the vector. Kaggle provides such an example. Imagine a straight line was recorded with 8 points. Then the simplified version removes 6 points to create a line with 2 points. Each csv file represents one "label" representing the drawing. Each row in the csv file represents a drawing or sketch with multiple strokes. Each stroke is represented by x,y, and t coordinates where t stands for time. Moreover, these are represented by nested arrays. 

<img width="241" alt="data" src="https://user-images.githubusercontent.com/39183226/49924634-b9628f00-fe6b-11e8-855f-028761919324.PNG">


## Approach: 

### Issues and Challenges
**Getting Data onto GCE**  
Since our dataset was so big (~70 GB total), it was hard to get it onto our google compute instance. We looked into and tried out a few approaches but the one we ended up using, which in our opinion was the simplest solution was to mount the file system of our GCE instance into our local machine and treat our entire GCE directory as a local subdirectory. We were then able to just copy the dataset as you would any file from one subdirectory to another. This was also helpful when we later wanted to actually open and look at some of the csv files in our dataset since using vim while we were SSHed to look at a csv with thousands of lines was rather messy. The instructions we found and followed on how to mount filesystems can be found [here](https://www.cs.hmc.edu/~geoff/classes/hmc.cs105.201501/sshfs.html).   
**Cleaning Data**  
For our CNN model, we tried converting the stroke of the drawings into images and saving them into subdirectories for later training. Even though we tried different variations improving the efficiency of our code, it would take incredibly long to run. We later found an online implementation that used a few tricks to expedite the cleaning code (i.e. not writing out the images into actual files, using matplotlib instead of PIL or openCV.   
**Loading and Cleaning the Data repeatedly for different runs**  
Since we were working with such a huge dataset, we would have to load and clean the data for every session. Even with the subset of the smaller dataset, this takes incredibly long. As such, we looked into ways we could save the loaded and cleaned panda dataframes for easier and faster reloading. We chose to use the HDF5 file format. However, even though we tried different ways of storing dataframes onto HDF5 files, we kept running into errors related to our version of python and or pandas. And since it did not seem reasonable to downgrade an already working part of our virtual environment, we chose to abandon this avenue.  
**Time and Resource Exhaustion during hyperparameter tuning**   
We repeatedly ran into Resource exhaustion errors. We would often not know how to fix this error so we would switch temporarily to a Kaggle kernel (considerably slower than our GCE instance). And since Kaggle has a 6 hour time limit and we also ran into the same error there, we concluded the error was not specific to our GCE instance. Upon further investigation, we found that our GPU memory was full. We fixed this by clearing our GPU memory using the %reset -f command. Note that clearing and restarting the kernel does not fix this.   
**Jupyter Lab crashing**   
We also had issues with Jupyter Lab where it kept crashing when we print out too much training information.

### RNN
The two most common type of neural network structures used were RNN and CNN. Our first choice was to implement an RNN. Consequently, we took a baseline model from Kevin Mader and decided to first see how well it performed. We discovered that there were many issues with the baseline model(523,002 trainable params). First, the training took too long, the accuracy was too low, and the model was too complicated for its subpar performance.  
We referred to various research articles and kaggle discussions on how we could improve the baseline RNN model. To begin with, we chose to simplify the model. The first thing we did was remove all dropout layers. Since the model was not doing well enough to even overfit, it did not make sense to have regularization yet. Removing all the dropout layers increased our initial accuracy of 4.6% and top 3 accuracy of 10.8% to an accuracy of 8.3 and top 3 accuracy of 15% (both for just 10 epochs on the whole data). This seemed to indicate we were on the right path so we proceeded to simplify the model even further. We removed all the 1D convolutions and instead used only one Global Average Pooling layer, which according to some experts in the kaggle discussion and a research paper found [here](http://arxiv.org/pdf/1312.4400.pdf) is actually better in that it can also be used to regularize. Through various combinations of parameters and after numerous test runs, we eventually chose to have the architecture shown below. However, there are various architectures that we came across during the course of this project that we came up with on our own or read about in publications.  
For instance, one of our simpler architectures with no dropout and just one LSTM (totally 44,396 trainable params) had accuracy: 13.0%, Top 3 Accuracy:25.3%.  
When we made the LSTM bidirectional we got Accuracy: 18.5%, Top 3 Accuracy 33.9%.
To put these values in context, the baseline model took 16.443 mins to train for 10 epochs on the whole dataset and got Accuracy: 4.6%, Top 3 Accuracy 10.8% whereas our model took 251 sec ~4.2 mins to train and had Accuracy: 18.5%, Top 3 Accuracy 33.9%. Also, please keep in mind that these accuracies are low because we are only doing 10 epochs and this was done to expedite hyperparameter tuning and training since having to run for hundreds of epochs everytime we change some parameter would take a few hours for a single model.  
Another interesting and effective thing we learned about over the course of this project is the keras callback method. Apparently, keras lets us specify a set of callbacks to keep in mind during training. These callbacks include but are not limited to reducing learning rate on a plateau, saving the model the second it does better (checkpoint),   and early stopping. In our case, reducing learning rate on plateau was particularly useful since our training plots from the baseline model show that there were lots of plateaus and we wanted to avoid these plateaus so that our model trains more efficiently. And so, we tinkered with the parameters of the ReduceLROnPlateau to avoid these plateaus and were pleased with the results. As is apparent from the explanations above and the graphs below, we avoided the plateaus, which resulted in a significantly shorter training time and even better accuracy. More detailed documentation of the keras' callbacks can be found [here](https://keras.io/callbacks/).


Initial RNN Results
![rnn_initial](https://user-images.githubusercontent.com/35898484/49917030-54e70600-fe52-11e8-868d-f5dbd7f3194a.PNG)

The Architecture of the modified RNN:   
<img width="306" alt="modified_rnn" src="https://user-images.githubusercontent.com/39183226/49924850-1fe7ad00-fe6c-11e8-86d3-b47cf18d084c.PNG">

Modified RNN Results:
![rnn_final](https://user-images.githubusercontent.com/35898484/49917041-65977c00-fe52-11e8-8c7d-da7a964f3e7a.PNG)


### CNN
We next attempted to compare the performance of our modified RNN to a CNN model since the model that won first place was a CNN model. As a result, we similarly took a baseline model from JohnM and after some minor tweaks. For the architecture of the model, we found that it was best to start with a 2D convolutional layer instead of a fully connected layer because it would otherwise lose spatial information. We also kept some of the original structure such as the implementation of max pooling to downsample the features and flatten to transform a tensor of any shape to a one-dimensional tensor. Finally, we incorporated dropouts in between each dense layer that we added. The reason we added it after a dense layer instead of a convolutional layer was that a convolutional layer has fewer parameters than a dense layer. As a result, a convolutional layer has less of a need for regularization to prevent overfitting compared to a fully connected dense layer. When ran on the Kaggle kernel, the final result of our implementation came out to be a validation accuracy of 65.63%, validation loss of 1.2015, and a top 3 validation accuracy of 85.17%. The results of the loss and accuracy are shown in the graph below.

Final Results: 
![cnn](https://user-images.githubusercontent.com/35898484/49917050-70521100-fe52-11e8-996f-dc249dda0dfc.PNG)

Modified CNN Architecture:   
<img width="241" alt="cnn_modified" src="https://user-images.githubusercontent.com/39183226/49924936-65a47580-fe6c-11e8-9f17-ac50221bda91.PNG">

## Thoughts on the difference between RNN and CNN
After comparing the results, we found that the CNN model resulted in a better accuracy compared to that of an RNN model. We began to think about the possible reasons why this was the case. In order to do so, we delved into the definitions of the models more deeply. We knew that the RNN was used to find patterns in sequential data. As a result, in this case, the RNN likely interpreted the drawing as sequences of 2D coordinates based on the individual strokes. The CNN on the other hand likely interpreted the drawing as a whole image with all the strokes at completion. Consequently, it would have interpreted the drawing as a 2D object. Understanding the differences in each model, we believed that one of the reasons the CNN performed better was the fact that each sketch depended on a variety of factors such as the nationality of the user, the individual person, and the speed of the sketch. Depending on the country or preference, many individuals have different starting points when they draw certain objects or letters. As a result, the model would need to be able to learn such differences. Consequently, when an RNN trains on a given data, it would need to account for the different ordering and speed of the individual strokes. Therefore, there may have been too many variations to generalize for the future data. However, intuitively, we believed that the CNN essentially "memorized" the image and therefore had better luck in generalizing for future data. 
RNN on the left and CNN on the right  
![rnn vs cnn](https://user-images.githubusercontent.com/35898484/50037232-0d808700-ffc4-11e8-8704-459162339a85.png)  
As can be surmised from the above plot, a CNN outperforms an RNN by a significant margin. However, this difference is probably significantly exaggerated here. It is possible and even highly likely that the CNN model we found could have already been optimized for increased performance. However, we believe that even if we had found equally simple RNN and CNN models, that the CNN model would have an even shorter training time and higher accuracy. It would be interesting to see how an LSTM+CNN model would perform given that the lack of convolutions in our final model is probably what makes the CNN significantly better.


## Time log
We have recorded performance for 16 different RNN models (they are mostly similar except for some minor changes) and each of these 16 initial models took about 16 mins each. As for our later simpler RNN models, we only recorded the runtime for 7 of them each about 4-5 mins. Note that these are just the ones we remember and recorded the runtimes for. There were many more intermediate models.   
In general, the overall runtime just for the RNN models with 10 epochs is about 16*16 + 7*4=284 mins~4.7 hours.  
As for the very few times, we ran a few of the better models on the whole dataset for a 100 epochs, that typically took about 
45 mins to an hour per model depending on the batch size and model (CNN or RNN).  
We spent considerable time looking into research publication and related Kaggle competition discussion and reading up on ways to improve performance.  
Also the above times are only runtimes for training the model and do not account for the time it took to load and clean the data.  
## Example Submission to Kaggle   
RNN baseline submission  
![rnn_baseline_submission](https://user-images.githubusercontent.com/35898484/50037183-9d720100-ffc3-11e8-8ea4-1b15060b480c.png)  
RNN Modified submission    
![rnn_modified_submission](https://user-images.githubusercontent.com/35898484/50037193-ba0e3900-ffc3-11e8-9a5c-1f3e646ca195.png)  

# References
Baseline RNN model forked from Kevin Mader [here](https://www.kaggle.com/kmader/quickdraw-baseline-lstm-reading-and-submission)  
CNN model forked from JohnM [here](https://www.kaggle.com/jpmiller/image-based-cnn)  
http://cs230.stanford.edu/files_winter_2018/projects/6921313.pdf  
https://uu.diva-portal.org/smash/get/diva2:1218490/FULLTEXT01.pdf  
https://arxiv.org/pdf/1704.03477.pdf  
https://www.theverge.com/2017/6/26/15877020/google-ai-experiment-sketch-rnn-doodles-quick-draw  
https://github.com/KKeishiro/Sketch-RNN/blob/master/sketch_rnn_model.py  
https://www.kaggle.com/c/quickdraw-doodle-recognition/discussion/70558  

