# Whale_Classification

https://www.kaggle.com/c/humpback-whale-identification

혹등고래의 꼬리이미지를 통해 고래의 종을 구분해 내는 competition 입니다.



# Overview

해당 고래 이미지셋을 분류하기 위해 Convolutional Neural Network을 활용하였습니다.

CNN 기법 중, 2017 이미지넷 챌린지(ILSVRC 2017)에서 우승한 SENet을 Resnet의 변형인 Resnext-101에 적용시킨 알고리즘을 활용하였습니다.


# SENet

![senet](https://user-images.githubusercontent.com/50981989/89257459-2807e800-d661-11ea-9466-3de1efdb3abc.PNG)

SE는 각 피쳐맵에 대한 전체 정보를 요약하는 Squeeze operation, 이를 통해 각 피쳐맵의 중요도를 스케일해주는 excitation operation으로 이루어져 있습니다. 이렇게 하나의 덩어리를 SE block이라고 합니다. 

SEnet은 네트워크 어떤 곳이라도 바로 붙일 수 있습니다. VGG, GoogLeNet, ResNet 등 어느 네트워크에도 바로 부착이 가능합니다.
파라미터의 증가량에 비해 모델 성능 향상도가 매우 큽니다. 이는 모델 복잡도(Model complexity)와 계산 복잡도(computational burden)이 크게 증가하지 않다는 장점이 있습니다.

![seres](https://user-images.githubusercontent.com/50981989/89257642-892fbb80-d661-11ea-90cd-dc4360455d95.PNG)

Senet을 Residual 모듈 뒤에 붙여 활용하는 방식입니다.

# Reference
 https://arxiv.org/pdf/1709.01507.pdf
 
 https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/models/senet.py
 
 https://jayhey.github.io/deep%20learning/2018/07/18/SENet/
