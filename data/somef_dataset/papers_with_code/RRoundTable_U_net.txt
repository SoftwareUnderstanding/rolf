# U_net
u_net for semantic segmentation


# model architecture

기존의 U_net구조에 pyramid구조를 더하여 global contex를 더 잘 추출해낼 수 있도록 만들었습니다.

결과적으로 기존의 U_net이 잡지못하는 작은 feature들에 더 좋은 성능을 보이고 있습니다.

### 기본구조
![u_net](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png)

### 추가한 구조 : pyramid

![pyramid network](https://hszhao.github.io/projects/pspnet/figures/pspnet.png)

reference : https://arxiv.org/pdf/1612.01105.pdf


# Data
https://www.kaggle.com/c/data-science-bowl-2018

# changes

- module화
- save result image


# result

## U_net
![train](./result/training_sample_300.png)

![validation](./result/validation_sample_15.png)

![validation](./result/validation_sample_35.png)

- 작은 feature는 잘 못잡아내고 있다.

## U_net + pyramid
아래의 이미지는 작은 feature를 상대적으로 잘 잡아내고 있다.
![pyramid](./result/validation_sample_35_pyramid.png)

# reference
https://www.kaggle.com/keegil/keras-u-net-starter-lb-0-277
