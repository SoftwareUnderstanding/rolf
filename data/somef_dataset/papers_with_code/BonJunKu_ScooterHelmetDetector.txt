# Scooter Helmet Detector

![sample](https://user-images.githubusercontent.com/62752488/122856946-a9886780-d352-11eb-9e1b-c46583045bec.gif)

# Demo Video
https://www.youtube.com/watch?v=ijOh_TxSuAE&ab_channel=%EC%9E%90%EB%9D%BC%EC%9E%90

# 프로젝트 보고서

https://drive.google.com/file/d/1eWKvClb_WRCqakboFkjekn0jRkhiEdJD/view?usp=sharing

# GOAL
헬멧 미착용 감지 및 경고 시스템 구현

# 연구 동기
 
법적으로 헬멧 착용이 의무화되므로, 효율적 단속과 안전 의식 확대를 위해 자동화 과정 개발이 필요하다고 요약할 수 있다.

# 아이디어 스케치

정거장 설치보다 주행 중 감시가 비용 면에서 효과적인데, 킥보드는 번호판이 없어 식별이 불가능하다.
그러나 킥고잉 사에 문의 결과, 이동 경로 및 장소/시간에 따른 조회가 가능함을 알았다. 따라서 아래와 같은 메커니즘으로 킥보드를 단속할 수 있다.

![image](https://user-images.githubusercontent.com/62752488/117592707-92abfe00-b174-11eb-8c7e-670f5b137b4d.png)


# 프로젝트 진행 경과


2021.03.02 ~ 2021.03.09 팀 빌딩 및 연구실 컨택

2021.03.09 ~ 2021.03.14 Object detection 관련 공부, 연구 계획서 제출

2021.03.17 ~ 2021.03.21 1차 멘토링

2021.03.22 2차 멘토링

2021.03.22 ~ 2021.04.03 환경 구축 및 오픈 소스 탐색, data set 확보, 학습

2021.04.03. ~ 2021.04.16 data set 재구성 및 재학습

2021.04.16.~2021.04.26 중간고사 기간, 중간보고서 작성 및 제출

2021.04.27.~2021.05.03 Data Augmentation을 통한 모델 성능 향상

2021.05.03.~2021.05.23 PyQt 공부 및 GUI 프로그래밍

2021.05.23.~2021.06.07 GUI 프로그래밍 마무리 및 데모 영상 제작

2021.06.07.~2021.06.22 기말고사 기간, 최종보고서 작성 및 제출



# reference
Zhengxia Zou et al.(2019). Object Detection in 20 Years: A Survey, IEEE, https://arxiv.org/pdf/1905.05055.pdf

Joseph Redmon et al.(2016). You Only Look Once: Unified, Real-Time Object Detection. IEEE, https://arxiv.org/pdf/1506.02640.pdf

oseph Redmon et al.(2018). YOLOv3: An Incremental Improvement. IEEE,https://arxiv.org/pdf/1804.02767.pdf

keras-YOLO3 https://github.com/qqwweee/keras-yolo3

OIDv4_Toolkit https://github.com/EscVM/OIDv4_ToolKit


# 환경설정
Windows 10 64-bit

Python 3.7.10

conda 4.9.2

TensorFlow GPU 1.15

CUDA Toolkit 11.2 

NVIDIA Geforce RTX 2080

NVIDIA GPU drivers 461.33

cuDNN 8.1.1

1. https://theorydb.github.io/dev/2020/02/14/dev-dl-setting-local-python/

2. https://raw.githubusercontent.com/chulminkw/DLCV/master/colab_tf115_modify_files/__init__.py

