# YOLOv1_implement_using_Tensorflow_or_Pytorch(텐서플로우, 파이토치를 이용한 YOLO구현)

<img width="618" alt="YOLO_architecture" src="https://user-images.githubusercontent.com/50979281/130927332-1aefef43-c67e-48db-98fe-68cd0a1ad629.png">


[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FCUAI-CAU%2FYOLOv1_implement_using_Tensorflow_or_Pytorch&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)


[텐서플로우](https://www.tensorflow.org)와 [파이토치](https://pytorch.org)를 이용해 YOLO의 최초 버전인 YOLO v1을 구현하며 객체탐지 모델이 어떤 과정을 거쳐 객체를 탐지하는지 알아보는 시간을 가졌습니다. 


YOLO v1 : https://arxiv.org/abs/1506.02640

## Members

[강민기](https://github.com/bbx8216)(School of Computer Science and Engineering, Chung-Ang University)
<br>
[김민규](https://github.com/MinkyuKim26)(School of Electrical and Electronics Engineering)
<br>
[김태윤](https://github.com/KimTaeYun02)(School of Computer Science and Engineering, Chung-Ang University)
<br>
[이승연](https://github.com/tmddus2)(School of Computer Science and Engineering, Chung-Ang University)

## Short paper
[여기](https://drive.google.com/file/d/1g2HjpD63xDjBhBL8mJzcx-P33yiGIe2L/view?usp=sharing)서 확인하실 수 있습니다

## How to implement

### Dataset

학습, 테스트를 위한 데이터셋으로 [PASCAL VOC 2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/) 을 사용했습니다. PASCAL VOC 2007은 학습용, 테스트용으로 데이터셋을 나눴는데 학습용 데이터셋의 양을 보존하기 위해 테스트용 데이터셋의 일부를 떼어 검증용 데이터셋으로 만들어 학습에 사용했습니다.


### Using Tensorlfow - Minki Kang, Minkyu Kim

 원래 Backbone으로 쓰던 [DarkNet](https://pjreddie.com/darknet/)이 Tensorflow로 구현된게 없었기 때문에 VGG와 DenseNet을 Backbone으로 사용했습니다. 학습을 위한 learning rate, momentum, weight decay는 논문에 있던 수치를 그대로 사용했고 여기에 [ModelCheckpoint](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint)를 사용해 훈련 도중에 Verification loss가 줄어들 경우 학습된 가중치를 저장해 훈련 중 가장 낮은 Verification loss를 가진 YOLO 최종 모델로 사용할 수 있게 만들었습니다. 
그리고 구현의 어려움으로 인해 Data augmentation을 구현하지 못했습니다.


#### test result
 
 ![test image](https://user-images.githubusercontent.com/50979281/131445170-794469fb-2c6c-434b-bb60-4e3e216b0119.png)
 
 학습용 데이터셋과 테스트용 데이터셋을 이용해 구현한 YOLO를 테스트 해본 결과, 학습용 데이터셋에서는 좋은 결과를 보여줬지만 테스트용 데이터셋에서는 좋은 결과를 보여주지 못했습니다. 오버피팅(Overfitting)이 일어난겁니다. 오버피팅이 일어난 이유로 Data augmentation을 구현하지 않아 다양하게 구성된 데이터셋에서 학습받지 못해서 일어난 것이라 판단하고 있습니다.


### Using Pytorch - Taeyun Kim, Seungyeon Lee
 
  파이토치를 통해 구현해보는 것이 처음이었기 때문에 이미 구현된 코드(https://github.com/aladdinpersson/Machine-Learning-Collection)를 보며 각주를 달고 이해하는 활동을 하였습니다. 학습시키는 과정에서 DataLoader 중에서 num_workers 관련 에러가 발생하여 YOLO 최종 모델을 구현하지는 못했습니다.
 


## How to test our model

Tensorflow : 'YOLOv1_implement_using_Tensorflow_or_Pytorch/MinKyu Kim/'에 있는 YOLO_test.ipynb과  [여기](https://drive.google.com/file/d/18wl62z2sU3O6NUl45K7iYSzWnGlpUYzV/view?usp=sharing)에 있는 yolo-minkyuKim.h5를 다운받아 같은 경로에 놔둡니다. 그리고 YOLO_test.ipynb를 실행한 뒤 코드를 위에서부터 차례대로 실행시킵니다.


Pytorch : 학습과정에서 오류가 발생해 테스트를 할 수 없습니다


