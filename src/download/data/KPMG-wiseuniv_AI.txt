## Fitting Room by 슬기로운 대학생활
안녕하십니까, 
저희는 COVID-19 이후 ‘인테리어 모바일 커머스 시장’이 급속도로 성장하고 있는 가운데, AI 기술을 활용한 인테리어 상품 추천 플랫폼 ‘Fitting Room’을 선보이고자 하는 ‘슬기로운 대학생활’ 팀입니다. 
![슬기로운대학생활2](https://user-images.githubusercontent.com/78267876/108618219-8959a200-745f-11eb-8561-36f777985df7.jpg)
![슬기로운대학생활_프로토타입_업로드용](https://user-images.githubusercontent.com/78267876/108618017-eeac9380-745d-11eb-9541-fc04be86f74f.gif)
## Summary
COVID-19가 전 세계적인 영향을 미치며, 자유로운 외부 활동이 거의 불가능해지는 현상이 지구 전체에 발생하였습니다. 이로 인해 집 안에서 활동해야 하는 시간이 늘어나면서 외부 활동과 관련된 소비액이 자연스레 ‘집안 활동’과 관련된 항목으로 옮겨가는 현상이 발생하였습니다. 

국내외 가구 브랜드에서는 AR, VR 등 다양한 기술을 접목한 홈 스타일링 서비스를 선보이고 있지만, 여전히 공간 전반을 아우르는 홈 스타일링 서비스를 제공받기 위해서는 매장을 방문해야 한다는 치명적인 단점이 존재하는 상황입니다.

이에 저희 팀은 모바일 홈 스타일링 솔루션 ‘Fitting Room’을 제공하고자 합니다. 

서비스에 관한 자세한 내용은 "APP Repositories"의 Readme에 정리하였으며, 다음의 URL을 통해 확인해주시면 감사하겠습니다.

## About Service (URL)
https://github.com/KPMG-wiseuniv/App/blob/main/README.md

## 모델 설명
### 1-1 segmentation
원하는 인테리어 속에서 추천하고자 하는 가구를 삭제하기 위해 DeepLab V3를 이용하여 Segmentation을 진행하였습니다.

![deeplab_v3](deeplab3_paper.PNG)

Rethinking Atrous Convolution for Semantic Image Segmentation,
Liang-Chieh Chen, George Papandreou, Florian Schroff, Hartwig Adam,
,2017_arXiv
[[paper]](https://arxiv.org/abs/1706.05587)

데이터는 NYU V2 이미지와 PascalVOC2012에서 가구 이미지가 있는 것을 골라 학습에 사용하였습니다. 

![example](modern_chair_3325_image.png)
![segmentation_mask](modern_chair_3325.png)

### 1-2 inpainting
저희는 지워진 가구의 위치에 inpainting을 하기 위해 아래의 논문을 이용하여 학습을 진행하였습니다.

![partial_convolution](partial_convolution_layer.png)
Image Inpainting for Irregular Holes Using Partial Convolutions, Guilin Liu, Fitsum A. Reda, Kevin J. Shih, Ting-Chun Wang, Andrew Tao, Bryan Catanzaro
, 2018_arXiv
[[paper]](https://arxiv.org/abs/1804.07723)

이미지 전체에 비해 가구가 차지하는 영역이 크지 않으며, 비어있는 방 이미지에서의 variance가 크지 않았기 때문에 Non-learning Based 
Model인 해당 모델을 이용하였습니다. 비록 computing cost가 매우 크기 때문에 real time으로 정보를 얻어내는 것이 불가능하다는 단점이 
있지만 저희는 학습하기 전의 단계에서 preprocessing을 거치는 것이기 때문에 괜찮다고 판단하였습니다. 

![example](modern_chair_3325_image.png)
![segmentation_mask](modern_chair_3325.png)
![inpainting_image](modern_chair_3325_inpainting.png)

저희는 빈방 이미지에서 랜덤한 box를 생성하여 마스크로 사용하여 해당 모델을 학습하였습니다. 기존의 논문에서는 Partial Conv를 이용한 VGG모델을 
ImageNet 데이터를 이용해 Pretraining한 후, Loss를 계산하였습니다. 하지만 저희는 빈 방 이미지를 먼저 VGG Network로 학습한 후 Inpainting Model을
학습하였습니다. 

![example](modern_chair_3325_image.png)
![segmentation_mask](modern_chair_3325.png)
![example](modern_chair_3325_cv2_inpainting.png)

이미지의 결과가 좋지 않은 경우에는 opencv의 cv2.INPAINT_TELEA 함수를 이용하여 학습에 사용하였습니다. 경계부분에서 내부 영역을 향해 점진적으로
점을 채우는 방법으로 주위의 픽셀값에 가중치합을 계산하여 마스크의 픽셀값을 복원합니다. 저희는 가중치의 값을 3으로 하여 실행하였습니다. 해당 코드는
*empty_room/opencv_inpainting.py* 에서 확인하실 수 있습니다.



### 1-3 fr model(recommendation)
Resnet이나 Inception 모델에서 Feature Extraction을 통한 Transfer Learning을 진행하려고 했으나 모델의 크기가 무거워진다는 단점이 있었기 때문에 Mobilenet을 통해 학습을 진행하였습니다. 

![MobileNet_v3](fr_model.PNG)

Backbone으로는 MobileNet v3를 이용하였으며 추천하기 위한 카테고리를 총 4개지로 나누어 4개의 Linear Layer를 학습하였습니다. 학습을 위해 
Adam Optimizer와 Cross Enrtropy Loss를 이용하였으며, 4개의 Linear Layer에서 나온 Loss를 사용하였습니다.
학습에 사용된 이미지가 매우 적었으며, Imbalance한 경우가 많았기 때문에 Oversampling과 Augmentation을 사용하였습니다. 
적은 수의 데이터를 랜덤하게 추가하여 해당 class의 데이터 수를 같도록 하였으며, RandomAffine, RandomRotation, RandomVerticalFlip 등의 Augmentation을
이용하였습니다.

## dataset소개
### 1-1 segmentation
NYU V2 이미지의 경우 depth정보를 제거하고, Image와 Segmentation Mask 정보만을 이용하였습니다. Pascal VOC2012의 경우
사람이 포함되지 않은 Easy Data만을 이용하여 학습을 진행하였습니다. 해당 이미지를 parsing하기 위해 *NYU_V2/NYU_V2_DataLoader.py*를
사용하였습니다. 

### 1-2 inpainting
빈 방이 이미지를 얻기 위해 크롤링하였습니다. 해당 코드는 *Furniture/crawling code.py*를 응용하여 사용하였습니다.

### 1-3 fr model(recommendation)
해당 이미지 역시 크롤링하여 사용하였습니다. 해당 코드는 *Furniture/crawling code.py*를 사용하였습니다.

### 실행 파일 소개
```
#segmentation
python Furniture/segmentation_main --train False

#inpainting
python empty_room/inpainting_main --train False

#fr_model(recommendation)
python Kaggle/main --train False
```

### 사용한 버전 및 모듈 소개
torch (1.7.1)

torchvision (0.8.2)

Pillow (8.1.0)

h5py (3.1.0)

torchvision (0.8.2)

typing-extensions (3.7.4.3)

opencv-python (4.5.1.48)
