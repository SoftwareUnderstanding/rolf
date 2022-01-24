# 교차로 상황인지 신호등 (중앙서버)
 CNN 이미지 객체 인식으로 CCTV 영상에 보이는 차량 대 수를 파악한 뒤 통행량이 많은 쪽으로 신호를 길게 조절하거나, 교통 사고를 인식하여 관리자에게 알림을 주는 신호등 개발

## 사용 기술
- C/C++
- cURL, wiringPi (Raspberry Pi)
- CNN

## 학습 전 설정
1. 데이터 위치 설정
> data/img/*<br>
> data/obj.data<br>
> data/obj.names<br>
> data/train.txt<br>
> data/test.txt

2. yolo-obj.cfg 수정
- anchor 설정
> $ ./darknet calc_anchors

3. darknet 옵션 목록 (기본값)
- show (0)		: calc_anchor 모드에서 결과이미지를 봄
- thresh (24)	: 낮을 수록 사소한 물체도 결과로 표시함
- num_of_clusters (5): calc_anchor 모드
- final_width (13)	: calc_anchor 모드
- final_height (13)	: calc_anchor 모드
- gpus (0)
- clear (0)

## 학습
> $ ./darknet train darknet19_448.conv.23

> $ ./darknet train backup/yolo-obj.backup > log/log_xxxx_yymmdd.txt

## 검증
1. imagenet test<br>
> $ ./darknet valid backup/yolo-obj_1500.weights

2. 학습로그???<br>
> $ ./darknet recall backup/yolo-obj_1500.weights

3. mAP 측정<br>
> $ ./darknet map backup/yolo-obj_1500.weights

## 실행
> $ ./darknet test backup/yolo-obj_5200.weights 0

## 실행결과 이미지 위치
> data/result/*

## 상황 인지 교통 신호등 제어방법 (src_desc/server.c)
```
493 Line : test_detector

1. 클라이언트로부터 받은 이미지 분석

94 Line : get_detect_result( ….. ) 함수 실행

103~123 Line : YOLO로 각 방향의 이미지를 분석하여 이미지 내의 차량대수를 센다.
126~134 Line : 분석 결과 이미지를 저장한다.


2. 분석결과를 토대로 신호제어 알고리즘 실행

229 Line : traffic_normal_mode( …. ) 함수 실행

248 Line : 신호 남은 시간이 0이 될 때마다 다음 신호 계산

239,255,272,287301 Line :
EW_Max (동-서로 이동하는 차량 수), NS_Max (남-북으로 이동하는 차량 수)를 비교하여 다음 신호 길이 조절에 반영한다.
    Ex )
    남북 직진 신호 or 좌회전 신호일 때
        calc_time = 3 * (NS_Max - EW_Max); //추가 시간

    동서 직진 신호 or 좌회전 신호일 때
        calc_time = 3 * (EW_Max - NS_Max); //추가 시간

264,279,294,308 Line :
 next_light에 다음 신호 종류를 저장하고 cur_light를 노란불로 바꾼다.

320 Line : 노란불 신호시간이 끝나면 cur_light를 next_light에 저장된 신호로 바꾼다.

329 Line : 추가 시간 (calc_time)은 -5 ~ 10 사이의 값으로 제한하여 각 신호에 기본으로 주어지는 시간인 20초에 더해진다. (결과 신호시간 15초 ~ 30초)

340 Line : 사고여부는 1 에서 이미지 분석할 때 동/서/남/북 중 하나 이상의 이미지에서 사고가 인식되면 웹서버로 사고 사실을 보낸다.
```

## 프로젝트 관련 링크
- [웹 서버 소스](https://github.com/dragon20002/STLCWebServer)
- 시연 영상<br>
[![시연영상](https://postfiles.pstatic.net/MjAxOTAyMThfMjc0/MDAxNTUwNDcwNzY4MDc3.1pafm9ZHEBZrCbK1ASPByV1ymMGSUUe-L9018VK05M0g.17RRvLqVNj_HIW4bfzsfFVQJklK_CR6RUmxklh91dokg.PNG.dragon20002/%EA%B7%B8%EB%A6%BC6.png?type=w580)](https://youtu.be/lfkUsUylsjE)
