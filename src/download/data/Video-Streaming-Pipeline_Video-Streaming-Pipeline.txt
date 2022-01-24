# Video-Streaming-Pipeline
실시간 이미지 처리 모델을 위한 모바일, 클라우드 영상 전송 파이프라인 개발

## 개발 동기
근래의 이미지 처리CNN 모델(MobileNet 제외)은 컴퓨팅 자원을 많이 요구하기 때문에 모바일이나 마이크로 디바이스에서 요구시간 안에 사용하기 힘들다.<sup>[1](#1)</sup><sup>[3](#3)</sup><sup>[4](#4)</sup> 그리고 MobileNet의 경우 지연율이 낮지만 다른 모델보다 정확도가 낮다.<sup>[2](#2)</sup><sup>[3](#3)</sup> 결국 고품질의 정확도와 짧은 처리시간을 모두 얻기 위해선 서버에 영상을 전송하여 처리한 후 받아와야 하므로, 단말에서 서버에 실시간으로 빠르게 영상을 전송하는 프레임워크가 필요하다. 또한 5G의 저지연율을 이용하면 단말에서 수행하는 것과 유사한 결과를 얻을 수 있을 가능성이 있다.

## 목표
1. 모바일에서 클라우드 모델로 영상을 실시간으로 전송하는 파이프라인을 개발한다.
2. 빠른 네트워크(5G 등)를 이용해서 클라우드 기반 모델이 모바일 기반 모델(MobileNet)만큼 빠른 속도로 서비스를 제공 할 수 있음을 증명한다.

## 구성도
<p align="center">
  <img src="https://user-images.githubusercontent.com/30094719/112486081-e9c16380-8dbe-11eb-8c36-5fb4addd8366.png" />
</p>

## 주요일정
[Notion 링크](https://www.notion.so/aff58c9b6b9e42f489443e14fd72606d)

# Usage
```
python benchmark.py
```
- 로컬에서 여러 모델을 실행시 평균 추론시간을 계산 할 수 있다.
- 기본적으로 CPU를 사용하며 GPU를 사용할 수 있는 환경인 경우 GPU를 사용하여 연산한다.

```
python benchmark_with_streaming.py
```
- 스트리밍을 통해 입력받은 모델에서 객체 탐지를 수행한다. -> 개선된 모델로 추가 예정
- 스트리밍시 프레임당 추론시간을 출력한다. (평균 추론 시간을 구하고 싶다면 benchmark.py를 사용하도록 한다.)
- 현재 데모 영상을 사용하고 있으며 라즈베리파이에서 수신하기 위해서는 VideoCapture를 다음과 같이 변경해야한다.

현재
```python
cap = cv.VideoCapture('https://www.freedesktop.org/software/gstreamer-sdk/data/media/sintel_trailer-480p.webm')
```

변경
```python
cap = cv.VideoCapture('udpsrc port=9777 ! application/x-rtp ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink', cv.CAP_GSTREAMER)
```
- 라즈베리파이와 연동시 네트워크 에뮬레이터를 통해서 4G 또는 5G의 환경을 조성해서 실행하도록한다.
- 서버에서 다시 라즈베리파이로 영상을 보내는 경우는 아직 구현되어 있지 않다.


## Reference
 <a name="1">1</a>. Table 1. Inference performance results from Jetson Nano, Raspberry Pi 3, Intel Neural Compute Stick 2, and Google Edge TPU Coral Dev Board, https://developer.nvidia.com/embedded/jetson-nano-dl-inference-benchmarks<br>
 <a name="2">2</a>. Image Classification on ImageNet, https://paperswithcode.com/sota/image-classification-on-imagenet<br>
 <a name="3">3</a>. Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam, _MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications_, https://arxiv.org/abs/1704.04861<br>
 <a name="4">4</a>. Andrey Ignatov, Radu Timofte, William Chou, Ke Wang, Max Wu, Tim Hartley, Luc Van Gool, _AI Benchmark: Running Deep Neural Networks
on Android Smartphones_ , https://arxiv.org/pdf/1810.01109.pdf
