# WeCare_makeathon

In this makeathon from Tum.ai i was allowed to work on the social impact challenge from google.
Here i had to build a system to detect people who are smokeing or going to smoke in public places like a petrol station

My Approach:
I wanted to create a human action recogntion algorithm that can classify people that are smoking. The idea is that it is often hard to detect smoke or cigarettes by themselves and its maybe easier to detect patterns of human movements.
Due to the lack of data i decided to use a dataset like Kinetics or Hmdb51.
Here i wanted to use a 3D CNN network as base model since 3D CNN can also preserve temporal information as stated in [[1]]

For my model i used as base the model described in [[3]]

  - the model consists of a pretrained base model from pytorch called r3d_18 [[1]]
  
  - i adjusted it to do only binary classification (smokeing or not)

  - i trained it on a subset of the Hmdb51 dataset with a weighted error to focus more on positive examples and overcome the unbalance between positive and negative    samples 
  
  - for negative samples i used 4 different actions (brush_hair, draw_sword, catch, eat)

  - the current model has an accuracy of 77% in testing 

  - the recall is quite high with 1.0 (which was initially intended to detect each person who smokes and rather more False Postives than False Negatives) therefore it makes sence to reduce the weights of the positive examples to increase the precision

To proceed:

  - a better suited dataset which films people smoking at a petrol station

  - extend the model to do Human Action Prediction we can achieve a better modelling of the temporal information by adding a LSTM after the r3d_18 [[2]]

  - also one can think about reducing the model size since the task of binary classification could be seen as easier. This could also boost in performance when it comes to real time applications. A simpler apporach could be the one stated in [[2]]

  - dataset to train a model that predicts a smokeing action
    (one could use the current algorithm and apply it on video data -> once it classify the frame as positive one can use the frames in prior before the peron starts to smoke)
    
[1]:  https://arxiv.org/abs/1711.11248  
[2]:  https://link.springer.com/content/pdf/10.1007/s42979-020-00293-x.pdf
[3]:  https://medium.com/@sauravsharma_19630/action-recognition-with-inbuilt-pytorch-features-ac2280d1c338

    
