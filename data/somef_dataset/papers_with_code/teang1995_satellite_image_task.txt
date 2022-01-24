#융합전자연구 기록
융합전자연구를 진행하는 과정에 대해 기록하는 repo입니다.

***
2020.01.21

dataset선정, 및 google drive 업로드. 

class별로 정리돼있는 폴더에 Train,Validation,Test 폴더 만들고 각각 560,70,70장으로 분류.
***
2020.01.24

dataset 폴더에 31500,3150,3150장씩(클래스 당 560,70,70) 나누어 저장 완료.

vggnet 이용한 classification 초안 업로드.

csv파일에 대한 질문 필요.

***
2020.01.27

vggnet 이용한 classification code 완성.

google drive의 시간 초과 오류 발생. 

해결 필요함.

***
2020.02.04

vggnet -> ResNet18로 모델 변경해야 함.

imagenet pretrain vs from scratch accuracy 비교해야 함.

data augmentation 추가해야 함

***
2020.02.11

ResNet18로 모델 변경 완료.

학습 되는 것 확인.

validation loss증가 시 weight 갱신하는 조건 더 명확히 해야 함.

질문 후 답변 기다리는 중.
***
2020.02.18

ResNet18 baseline : 50epoch에서 79~80%의 Accuracy 보임.

ResNet18 baseline에 다음의 trick들을 도입함.

  - learning rate warmup
  
  - cosine learning rate decay
  
  - ResNet tweak C,D
 
ResNet18 with tricks : 70epoch정도에서 88~90%의 Accuracy 보임.
  
다음의 trick들을 추가 도입해야 함.

  -label smoothing
  
  - data mixup(https://arxiv.org/pdf/1710.09412.pdf)
  
  - random scaling crop , horizontal flipflop.

어떤 trick의 영향력이 가장 큰지 개별적으로 실험 필요함.

SSD구현, detection에 적합한 1GB 이내 용량의 dataset 조사 필요함. 

