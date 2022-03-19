# Scooter Helmet Detector

![sample](https://user-images.githubusercontent.com/62752488/122856946-a9886780-d352-11eb-9e1b-c46583045bec.gif)

# Introduction

- 전동킥보드 헬멧 착용여부를 감시하는 프로그램입니다.

# Demo Video

- 아래 YouTube 영상을 참고해 주시기 바랍니다.

- https://www.youtube.com/watch?v=LFBeaK554nc

# 프로젝트 보고서

- 자세한 프로젝트의 진행내역은 아래 링크의 보고서 안에 기록되어 있습니다.

- https://drive.google.com/file/d/1eWKvClb_WRCqakboFkjekn0jRkhiEdJD/view?usp=sharing

# 실행 방법

1. `env.yml`파일로 가상 환경을 설치합니다.

```
conda env create -f env.yml
```

2. 가상환경을 activate합니다.

```
conda activate tf1
```

3. `Scooter_Helmet_Detector.ipynb`에서 `HOME_DIR`를 사용자 환경에 맞게 설정한 후, 모든 라인을 실행해줍니다. (원활한 동작을 위해서는 고성능 GPU가 필요합니다.)

# Reference

- Zhengxia Zou et al.(2019). Object Detection in 20 Years: A Survey, IEEE, https://arxiv.org/pdf/1905.05055.pdf
- Joseph Redmon et al.(2016). You Only Look Once: Unified, Real-Time Object Detection. IEEE, https://arxiv.org/pdf/1506.02640.pdf
- Joseph Redmon et al.(2018). YOLOv3: An Incremental Improvement. IEEE,https://arxiv.org/pdf/1804.02767.pdf
- keras-YOLO3 https://github.com/qqwweee/keras-yolo3
- OIDv4_Toolkit https://github.com/EscVM/OIDv4_ToolKit

# Environments

- Windows 10 64-bit
- Python 3.7.10
- conda 4.9.2
- TensorFlow GPU 1.15
- CUDA Toolkit 11.2
- NVIDIA Geforce RTX 2080
- NVIDIA GPU drivers 461.33
- cuDNN 8.1.1
- 환경설정을 위한 참고의 글
  1. https://theorydb.github.io/dev/2020/02/14/dev-dl-setting-local-python/
  2. https://raw.githubusercontent.com/chulminkw/DLCV/master/colab_tf115_modify_files/__init__.py
