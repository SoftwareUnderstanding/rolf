# Graduate-Paper
2020.02 통계학 석사 졸업 예정

## Session-Based Recommendation with RNN
### 참고 논문
1. [***Session-Based Recommendations with Recurrent Nerual Network***](https://arxiv.org/pdf/1511.06939.pdf) (original Paper)
Balázs et al., ICLR, 2016.
- RS에 RNN 적용
- https://github.com/hidasib/GRU4Rec (original code) : theano로 구현, RecSys2015
- https://github.com/Songweiping/GRU4Rec_TensorFlow : theano -> tensorlow implementation
- https://github.com/khlee88/GRU4Rec_tutorial : tensorflow tutorial
- https://github.com/pcerdam/KerasGRU4Rec : keras ver
- https://github.com/yhs-968/pyGRU4REC : pyTorch ver
  
2. [***Improving Session Recommendation with Recurrent Neural Networks by Exploiting Dwell Time***](https://arxiv.org/pdf/1706.10231.pdf)
Alexander et al., 2017
- 

3.[***Incorporating Dwell Time in Session-Based Recommendations with Recurrent Nerual Networks***](http://ceur-ws.org/Vol-1922/paper11.pdf)
Veronika et al., RecSys2017
- 1번 논문 방법에 dwelling time만 적용하여 비교한 논문
- 상당한 차이 있음.
- dwelling time에 따라 세션 수 를 증가시켜 

4.[***Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation***](https://arxiv.org/pdf/1406.1078.pdf) Cho et al., arXiv:1406.1078
- GRU paper


### 과정 정리
- [2019-08-23] 교수님께 논문 방향 컨펌. 모델 설명 잘 정리할 수 있을지..?
- [2019-08-24] keras 소스코드 이용해서 모델 돌아가는 거 확인!(GPU 사용, 1epoch : 1시간 30분정도 걸림), class를 이용해서 데이터 로드하는 거까지 대강 파악! SessionDataLoader의 iter를 이용하여 yield되는 **inp, target, mask** 원리 파악이 더 필요함.
- [2019-09-01] 한 epoch당 90분 정도 걸림. GPU가 잘 도는지 모르겠지만,, (epoch 10번).(negative sampling하지 않은 keras 코드 사용). dwelling time 이용하여 session수 늘리는 건 어렵지 않을 듯
- [2019-09-02] 데이터 300만개로 원래 데이터와 dwelling time augment한 데이터(threshold = 200000)로 5epoch으로(epoch당 약 15분 소요) 모델링. 결과는 큰 차이 없음
- [2019-09-03] rsc15랑 rsc19데이터 모두 비교

### 논문 방향
- 동일 데이터로 파라미터 조정(lr, batch_size, optimizer 등등) 기존 논문 beat하기?
- 동일 데이터로 negative sampling하지 않은 거랑 비교?
- 19데이터 이용하여 dwelling time 적용.

