# med_TwoCycleLearning-and-MixUp
This is a project which use two cycle learning and mix up to try to improve model performance when there are few labeled data while there is a lot of unlabeled data. We use MURA (https://stanfordmlgroup.github.io/competitions/mura/) as our dataset, to simulate the few-labeled-data situation, we randomly pick data and their label with a paramter (number of labeled data / number of all data), and the remainder we discard their label.

We chose Two Cycle Learning  (https://arxiv.org/abs/2001.05317) as our semi-supervise learning algorithm, and mixup (https://arxiv.org/pdf/1710.09412.pdf) as our data augmentation algorithm.

Below is the training flow of our implementation of two cycle learning.
![model_flow](2cl.png)
