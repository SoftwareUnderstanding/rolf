# NuguEyeTest

----
## 1. 프로젝트 개요
    누구 네모를 위한 플레이를 개발하는 것이 목적인 프로젝트
    기존 누구의 AI speaker만을 이용하는 것이 아닌 카메라와 디스플레이가
    존재하는 NUGU nemo에 적용할만한 아이디어를 고안하고
    이를 실제로 구현해보고자 함.


---- 
## 2. 구성
1. [Nugu Play Builder](https://developers.nugu.co.kr/#/play/playBuilder?d=1582182375657)

2. [Google Cloud Function](https://cloud.google.com/functions) ( GoogleCloudFunction/CapInfo.js )

3. Relay server ( GoogleCloudVM/server.py )

4. Device python code ( eyeTestSSDPose.py )
    
----

### 2.1 Nugu Play Builder
* Nugu Play Builder는 Nugu에서 제공하는 GUI 형태의 play 개발 플랫폼이다. 
* GUI 형태여서 간단한 구조는 손쉽게 개발할 수 있다.

<img src="/NuguTree.png" width="100%" height="100%" title="px(픽셀) 크기 설정" alt="NuguTree"></img>

* 위와 같이 트리형태의 구조를 띄고 있다. 계층적인 구조이며 그에 맞게 발화 내용이 처리 된다.

* Nugu Play Builder에 대한 간단한 제작 방법이나 설명은 추후에 추가할 예정

### 2.2 Google Cloud Function
    Google Cloud Function 이란?
* Google cloud의 이벤트 기반 서버리스 컴퓨팅 플랫폼. 즉, 서버를 유지할 필요 없이 실행할 코드만 클라우드에 올려놓으면 특정 이벤트(put, get, … rest API 형태)가 생길 시에 코드가 실행되는 형태.
서버를 관리할 필요가 없이 손쉽게 서버처럼 이용 가능. ( invoke 된 횟수, 사용한 리소스만큼만 금액을 지불하면 됨. )

* GoogleCloudFunction directory에 어떤 식으로 코드를 첨부해야하는지에 대한 설명을 추가할 계획

### 2.3 Relay Server
* 원래는 Google Cloud Function 에서 바로 노트북에 socket으로 메세지를 보내서 누구 플레이의 발화를 전달하려고 하였으나 학교의 인터넷 환경상 공인 ip를 사용할 수 없었기 때문에 Relay Server를 두었다. ( Google Cloud Computing engine의 VM을 사용 )

* 원래는 이러한 가상 서버에서 이미지에 대한 처리를 하려 했으나 누구 네모에서의 카메라 제어나 파일 전송 등을 모두 지원하지 않아서 Relay Server의 기능과 몇 가지의 플레이 진행상황(status)을 다루게 하였다.

* 현재는 코드가 다수의 기기가 아닌 1개의 기기와의 연결만 지원하는 형태로 구현되어 있다. 추후에 작업을 통해 누구 디바이스에 따라서 메세지를 처리해주게 변경해야 한다.

### 2.4 Device python code

* 프로젝트의 메인인 부분이다. 크게 4가지로 구성되어 있다.

----
    1. Relay Server로부터 발화 명령을 전달받고 그에 따른 이미지를 화면에 나타내는 부분.
    2. 현재 카메라로 받는 이미지에서 얼굴 및 눈을 검출하는 부분.
    3. 한 장의 이미지에서 사용자의 카메라까지의 거리를 구하는 부분.
    4. 사용자의 자세를 추정하여 눈을 가린 여부를 확인하는 부분.

----

## Reference
### 1. Dlib: A Machine Learning Toolkit
http://www.jmlr.org/papers/volume10/king09a/king09a.pdf
https://github.com/davisking/dlib

facial landmark detection을 위해 Caffe로 학습한 모델

[“One Millisecond Face Alignment with an Ensemble of Regression Trees,” Vahid Kazemi and Josephine Sullivan, CVPR 2014.](http://www.nada.kth.se/~sullivan/Papers/Kazemi_cvpr14.pdf)

### 2.Single Shot MultiBox Detector 
https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector
Opencv 에서 사용할 수 있게 TensorFlow와 Caffe로 학습한 모델

[“Single Shot MultiBox Detector,” Wei Liu, Dragomir Anguelov, ECCV 2016.](https://arxiv.org/pdf/1512.02325.pdf)

https://github.com/weiliu89/caffe/tree/ssd 


### 3.[OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields](https://arxiv.org/abs/1812.08008)
https://github.com/CMU-Perceptual-Computing-Lab/openpose

https://arxiv.org/pdf/1611.08050.pdf

현재 사용한 모델은 2016년 논문에 제시된 모델을 caffe로 학습한 모델임


