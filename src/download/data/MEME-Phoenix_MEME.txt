[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FMEME-Phoenix%2FMEME&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

<h1 align="center">
  <br>
  <a href="http://www.amitmerchant.com/electron-markdownify"><img src="./logo.png" alt="Markdownify" width="400"></a>
  <br>
  Autonomous Driving Trolley, MEME
  <br>
</h1>

<h4 align="center">당신의 쇼핑 생활을 upgrade해 줄 <a href="https://www.notion.so/Autonomous-Driving-Trolley-MEME-01fdd602990b4baa9b603d419a1479bb" target="_blank">MEME</a>.</h4>

<p align="center">
  <a href="#overview">Project Overview</a> •
  <a href="#use">How To Use</a> •
  <a href="#who">Contributors & Blogs</a> •
  <a href="#ref">References</a> •
  <a href="#sang">Awards & Patents</a> •
  <a href="#lics">License</a> 
</p>

<p align="center">
  <a href="https://youtu.be/U72QXuIItQo">Presentation Video</a> •
  <a href="https://drive.google.com/file/d/1sCxK6VSaGHfekCug8bh8UokbR4JPQqiN/view?usp=sharing">Paper</a> •
  <a href="https://drive.google.com/file/d/1EO5TITTPSDBDCRN48HDSJHg3zqUbV14D/view?usp=sharing">Poster</a> •
  <a href="https://bit.ly/3lN3iEF">Notion</a> •
  <a href="https://github.com/MEME-Phoenix/Autonomous-Driving-Cart-MEME">GitHub</a>
</p>
<div id = "overview">

## ✨ Project Overview

### ‼️ Key Features

1. Object Tracking System with YOLOv5 & DeepSORT
2. Keyword Spotting: RNN model on word "미미야"
3. Realtime Location Track with LiDAR Sensor
4. Emergency Detection with ultrasonic Sensor
5. Embedding System

### ⚒️ System Architecture

<p align="center" >
<img src="./system_architecture_eng.png" alt="Markdownify" width="800" >
</p>
</div>

<div id = "use">

## 🛎️ Quick Start

본 시스템은 Ubuntu 18.04, ROS Melodic 환경에서 작동됩니다. 다음과 같은 라이브러리 설치가 필요합니다. 

- Python 3버전 이상
- numpy
- scipy
- opencv-python
- sklearn
- torch ≥ 1.7
- torchvision ≥ 0.1
- pillow
- vizer
- edict

---

1. Clone this repository & install requirements

```bash
$ git clone https://github.com/MEME-Phoenix/MEME.git
$ pip install -U -r requirements.txt #위 requirements 설치
```

2. Download YOLOv3 parameter

```bash
$ cd detector/YOLOv3/weight/
$ wget https://pjreddie.com/media/files/yolov3.weights
$ wget https://pjreddie.com/media/files/yolov3-tiny.weights
$ cd ../../../
```

3. Download DeepSORT parameters ckpt.t7

```bash
$ cd deep_sort/deep/checkpoint
# download ckpt.t7 from
https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6 to this folder
$ cd ../../../
```

4. Compile nms module

```bash
cd detector/YOLOv3/nms
sh build.sh
cd ../../..
```

5. Rum Demo

```bash
$ usage: python3 train.py VIDEO_PATH
                          [--source 0] #실시간
                          [--source [mp4 file name]] #영상 첨부시
```

Note: If you're using Linux Bash for Windows, [see this guide](https://www.howtogeek.com/261575/how-to-run-graphical-linux-desktop-applications-from-windows-10s-bash-shell/) or use node from the command prompt.
</div>

<div id = "who">

## 👥 Contributors & Blogs

- 한지수 [@JisuHann](https://github.com/JisuHann)
    - [[졸업프로젝트 개요, 1탄 RNN] 딥러닝을 이용한 자율주행카트](https://jisuhan.tistory.com/entry/졸업프로젝트딥러닝을-이용한-자율주행카트)
    - [[졸업프로젝트 2탄, CNN] ResNet50 톺아보기: 구조와 코드 분석](https://jisuhan.tistory.com/entry/CNN-ResNet50-톺아보기-구조와-코드-분석)
    - [[졸업프로젝트 3탄, HW] turtlebot3로 SLAM, Navigation 구현(2020 Summer)](https://jisuhan.tistory.com/entry/turtlebot3로-SLAM-Navigation-구현하기)
    - [[졸업프로젝트 4탄, 알고리즘 학습] AWS EC2로 딥러닝 모델 학습하기](https://jisuhan.tistory.com/entry/졸업프로젝트-4탄-알고리즘-연결-알고리즘-완성하기)
  - 그로쓰 학기
- 박지윤 [@jiyoonpark0207](https://github.com/jiyoonpark0207)
    - [[1탄] Yolo v3 를 이용한 인물 추적 프로젝트](https://yumissfortune.tistory.com/4)
    - [[2탄] Yolo v3 를 이용한 인물 추적 프로젝트](https://yumissfortune.tistory.com/5)
    - [[초보탈출] ROS 를 이용한 다양한 실습](https://yumissfortune.tistory.com/17)
- 김채원 [@cwkim0314](https://github.com/cwkim0314)
    - [[IT/KR/Project] 자율 주행 카트를 만들어보자](https://blog.naver.com/cwkim0314/222156573981)
    - [[IT/KR] Object Detection - EfficientDet](https://blog.naver.com/cwkim0314/222156584109)
    - [[IT/KR/Project] Hardware: Alphabot2-pi](https://blog.naver.com/cwkim0314/222167401417)
</div>

<div id = "ref">

## 🔭 References

1. Object Tracking
   - Object Tracking(https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch)
   - Simple Online and Realtime Tracking with a Deep Association Metric(https://arxiv.org/abs/1703.07402)
   - YOLOv4: Optimal Speed and Accuracy of Object Detection(https://arxiv.org/pdf/2004.10934.pdf)
2. Keyword Spotting
3. Embedding System

</div>

<div id = "sang">

## 🏆 Awards & Patents
- 2021 특허 출원 완료(P2021-0069KR)
- 2021 한국정보처리학회 논문 게재
- 2020 캡스톤경진대회 창업아이디어 부문 우수상
- 2021 캡스톤디자인프로젝트 포스터세션 최우수상
- 2021 SW인재페스티벌 이화여자대학교 대표로 참가, 42팀 중 인기상 수상 [메타버스 발표 영상](https://youtu.be/ndY08MRQkJY)
</div>

<div id = "lic">

## License

Copyright (c) 2021 MEME-Phoenix See the file license.txt for copying permission. LICENSE를 참고하세요.
</div>
