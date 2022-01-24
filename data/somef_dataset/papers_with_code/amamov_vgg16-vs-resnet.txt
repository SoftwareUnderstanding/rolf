# vgg16-vs-resnet project

> 기존의 전통적인 CNN 모델인 VGG16과 ResNet 리뷰 및 성능 비교 분석.


## Reference

- `VGG16` :
  - 논문 링크 : [https://arxiv.org/pdf/1409.1556.pdf](https://arxiv.org/pdf/1409.1556.pdf)
- `ResNet` :
  - 논문 링크 : [Deep Residual Learning for Image Recognition](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)

## Colabo Tool

- github
- notion, google docs
- google colab

## Framework

- CIFAR 10
- pytorch
- keras
- torchvision


## log

동일한 데이터 셋과 환경에서 VGG 모델과 ResNet 모델의 성능 차이는 약 `10%`만큼 차이 났다. 이와 더불어 위 실험 코드는 기본적인 CIFAR10 데이터 셋을 기반으로한 재사용가능한 VGG 모델 머신 클래스를 설계하고 ResNet 모델 머신에서 상속받아 batch_size, epoch_size, learning_rate, momentum 파라미터를 조정할 수 있도록 클래스 설계를 했다. 슈퍼 클래스인 VGG 모델 머신 클래스를 기반으로 ResNet 뿐만 아니라 다양한 모델 기반의 머신 클래스를 추가적으로 상속을 통해 설계할 수 있다. 해당 코드를 확장하여 추후에 다른 프로젝트에서 여러가지 다른 모델들을 비교 분석을 할 계획이다. 또한 하나의 모델에서 4개의 하이퍼 파라미터를 조정하면서 비교 분석도 할 계획이다.
