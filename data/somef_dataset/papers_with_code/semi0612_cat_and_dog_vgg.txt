###### 문제가 될 경우 삭제하겠습니다.
이 실험은 Tensorfloww Library API을 기반으로 Colab이 제공하는 GPU를 사용했습니다. 

## * cat_dog_vgg
#### - 데이터는 kaggle에서 제공하는 Cats-vs-Dogs 데이터를 이용하였습니다. 2500장 중 2000장을 train_data로 500장을 vaild_data로 나누어 진행하였습니다.
성능이 좋은 모델과 좋지않은 모델을 비교합니다.

## * Cats_Dogs_Grad_CAM
#### - 데이터는 kaggle에서 제공하는 Cats-vs-Dogs 데이터를 이용하였습니다. 2만 5천장 중 1만 7천장을 train_data, 4천장을 validation_data, test_data 4000으로 나누어 실험을 진행하였습니다.

### 실험 모델
VggNet(Layer 16, epoch 250, optimizer SGD(lr=0.001, momentum=0.9), batch_size 20, step(train, batch) 80, 20, loss function binary crossentropy, dense layer activation sigmoid)

Data Augmentation

    - train : rotation_range=40, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, rescale=1./255
    - validation : rescale=1./255
    - test : rescale=1./255

### 모델 학습 결과(성능)
Accuracy 95.05%, Loss 0.1351

### Grad-CAM
간단히 말해서, Grad CAM은 얼굴 위치 추적기라고 부르기도 함. 이 Grad-CAM을 사용하여 모델이 피쳐 맵이 주로 어디를 바라보고 분류하는지 확인하며 잘 못 예측한 경우 주로 어디를 바라보며 잘 못 예측한 것인지 알고자 함.

    - predict score 0.5이상이면 dog class 예측한 경우이며 0.5미만이면 cat class 예측한 것이라고 생각하면 됨. 


### 맞게 예측한 Grad-CAM

#### 고양이
![image](https://user-images.githubusercontent.com/45933225/83970905-3a8ccc00-a913-11ea-82ee-3f9326aa58f1.png)

#### 개
![image](https://user-images.githubusercontent.com/45933225/83971064-27c6c700-a914-11ea-80d6-8a226808c98d.png)

#### 이 모델에서 전반적으로 고양이를 맞게 예측하는 것은 콘트라스트한 이미지에 이목구비 또는 다리 전체적으로 바라보는 것이 아닌 특정 한 곳을 바라보며 예측하는 것이라 생각이 되며 개의 경우는 똑같이 콘트라스트한 이미지의 경우와 고양이와는 다르게 특정 한 곳이 아닌 이목구비 전체적으로 바라보며 예측한다고 생각이 됨.



### 잘 못 예측한 Grad-CAM
분석을 위하여 조금 비슷한 상황의 이미지들끼리 묶어서 구별하였음.

#### 고양이

- 배경에 선들이 그어져 있는 경우(철장, 벽돌, etc..)

![image](https://user-images.githubusercontent.com/45933225/83971265-804a9400-a915-11ea-8858-0173678edf6d.png)

- 사람하고 같이 있는 경우(성인 여성, 성인 남성, 유아, etc..)

![image](https://user-images.githubusercontent.com/45933225/83971531-acb2e000-a916-11ea-9f2c-be3e5503f038.png)

- 그 밖에 경우

![image](https://user-images.githubusercontent.com/45933225/83971696-bb4dc700-a917-11ea-99fe-62941dd26688.png)

주로 위의 경우들로 예측을 잘 못하는 경우들이 많았으며 그 밖에 배경에 글씨가 있는 경우, 개랑 같이 있는 경우, 다른 사물을 바라보는 경우, 고양이 이미지가 너무 작은 경우, 고양이 주변 커다란 타원이 있는 경우 등 여러가지의 상황들이 있었음.

#### 개

- 다리를 보는 경우

![image](https://user-images.githubusercontent.com/45933225/83972300-5647a080-a91a-11ea-8a33-6ad1ad418c7c.png)

- 주로 코만 보는 경우

![image](https://user-images.githubusercontent.com/45933225/83972525-8f344500-a91b-11ea-8b60-e1220b995823.png)

- 얼굴이 잘 안보이는 경우(다른 곳을 바라봄)

![image](https://user-images.githubusercontent.com/45933225/83972692-97d94b00-a91c-11ea-9785-576088dd891d.png)

- 콘트라스트가 낮은 이미지의 경우

![image](https://user-images.githubusercontent.com/45933225/83972921-f0f5ae80-a91d-11ea-974e-075e895cab11.png)

- 눈 주위를 보는 경우

![image](https://user-images.githubusercontent.com/45933225/83973101-08816700-a91f-11ea-9d12-194709610e86.png)

- 그 밖에 경우

![image](https://user-images.githubusercontent.com/45933225/83973373-f56f9680-a920-11ea-8b1c-ec503c6dad3b.png)

주로 위의 경우들로 예측을 잘 못하는 경우들이 많았으며 그 밖에 배경에 철장이 있는 경우, 글씨를 보는 경우, 귀를 보는 경우, 다른 사물을 보는 경우 여러가지 상황들이 있었음.

#### 주로 고양이보다는 개를 잘 못 예측하는 경우들이 많았음.

#### 고양이 같은 경우 주로 특정 한 곳을 주로 보며 예측하는 경우들이 많다. 그래서 위의 잘 못 예측하는 경우들을 확인하면 이미지에 잡음이 이미지, 다른 사물 or 사람하고 같이 있는 것을 볼 수 있다. 그래서 Grad-CAM이 그 곳을 바라보거나 못 찾는 경우들이 발생하는 것을 확인할 수 있었음.

#### 개의 경우는 주로 이목구비 전체적으로 보며 예측하는 경우들이 많다. 그래서 위의 잘 못 예측하는 경우들을 확인하면 이미지 특정 한 곳을 바라보는 경우(눈, 코, 귀), 잡음 이미지, 스무딩한 이미지, 다른 사물 or 사람하고 같이 있는 것을 볼 수 있다. 그래서 Grad-CAM이 그곳을 바라보거나 못 찾는 경우들이 발생하는 것을 확인할 수 있었음.
