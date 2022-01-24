# Melanoma_classification

https://www.kaggle.com/c/siim-isic-melanoma-classification

피부병변 이미지를 통해 흑색종을 구분해 내는 competition 입니다.

![순위2](https://user-images.githubusercontent.com/50981989/90606126-56c5b700-e23a-11ea-894a-ca9087674dce.PNG)

3319팀 중 378등을 하였습니다.



# Overview

https://paperswithcode.com/sota/image-classification-on-imagenet

![effi](https://user-images.githubusercontent.com/50981989/90606537-ed927380-e23a-11ea-9d47-f87eed77a08f.PNG)

2020년 8월 기준, Efficientnet기반의 모델인 FixEfficientnet-L2 가 SOTA(state-of-the-art) 이며 여러 efficient 기반 모델들이 성능이 좋은걸 알 수 있습니다.  

이번 분석에는 대중적으로 사용하고 있는 Efficientnet중 efficient-b1가 기반인 코드를 사용하였습니다.

efficient-b1~b4 를 ensemble 한 뒤, 결과를 여러 note들과 ensemble 하여 제출하였습니다.


# Efficient-net

![eff1](https://user-images.githubusercontent.com/50981989/90620382-5387f680-e24d-11ea-8f6f-363ad7d2de80.PNG)

논문의 연구결과를 보면 Efficientnet이 SEnet 과 Resnext 등의 모델보다 월등히 좋은것을 알 수 있습니다.

EfficientNet 논문은 ConvNets의 스케일을 높이는 기존 접근법을 다시 한번 짚고 넘어가자는 의미의 'Rethinking'을 논문의 타이틀로 사용하였습니다.

Effi-net은 모델 스케일링 기법을 중점으로 다뤘으며, 최적 Width-Depth-Resolution 을 찾는것이 주 목적입니다.



# Reference

https://arxiv.org/abs/1905.11946

https://www.kaggle.com/nroman/melanoma-pytorch-starter-efficientnet
