# Real or Not? NLP with Disaster Tweets(NLP)
- predict which tweets are about real disasters and which ones are not.
- 트위터는 긴급상황에서 중요한 커뮤니케이션 채널이다. 
- 스마트폰의 편재성 덕분에 사람들은 실시간으로 관찰중인
비상상황을 발표할 수가 있다. 이로 인해 많은 대행사가 프로그래밍 방식으로
Twitter를 모니터링하는데 관심이 많다. 
- 그러나 사람의 말이 실제로 재난을 알리는지 여부가 항상 명확한 것은 아니다 
- 실제 재난에 대한 트윗과 그렇지 않은 트윗을 예측하는 기계 학슴 모델을 구축하고자 한다. 

--------------------------------------------------------------------------
## model
- 특히 custumizing한 부분에 중점되어 설명
- 각 layer가 어떤 의미를 가지고 어떻게 변형했는지에 중점을 맞출 것
- feature은 자의적으로 바꾸지 말고 PCA와 같은 변수 선택할 수 있는 method를 사용할 것 

**(1) tweeter_disaster_baseline.ipynb**
- txt 데이터를 counter vectorize화 하여 분석하고자함
- Ridge Classifier, xgboost + tuning  classify

**(2) tweeter_disaster_2.ipynb**
- BERT ( https://arxiv.org/abs/1810.04805) 논문 읽고 구현
- using model named BERT
- Load BERT from the tensorlow hub
- load tokenizer from the bert layer
- encode the txt into tokens, masks, and segment flags
- **no pooling, directly use the CLS embedding.** 
- **no dense layer, simply add a sigmoid output directly**
- fixed parameters


--------------------------------------------------------------------------

## NLP model 

### 1.  RNN, LSTM
- 결국   LSTM은 RNN에 cell state를 추가한 형태이다. 

### 2. seq2seq2 
- 입력값을 받아 벡터 c^2로 변환해주는 encoder와 encoder에서 출력된 값을 입력으로 받아 conditional probability를 고려하는 decoder로 이루어 진다. 
- 하지만 입력 시퀀스가 매우 길 경우에는 long time dependencies 문제가 일어난다. 일반적으로는 LSTM이 이를 해결하나, 아직 문제가 존재한다. 
- 이 때문에 **입력 시퀀스를 뒤집어 디코더에서 인코더의 부분까지를 단축하여, 보다 좋은 성능을 보이기도 한다.**
(왜 뒤집으면 경로가 단축되지? : 전체 평균거리는 유사하나, 1대1로 매핑되었을 때의
거리가 가까워진다, 처음 값을 제외한 나머지 출력층의 추측의 경우는 seq특성상 앞쪽
5가 추측이 되면 short term dependancy를 이용하여 추측이 쉬워진다. 
따라서 속도나 정확도가 올라간다. )
(***이렇게 되면 문맥적인 학습이 잘못되지 않는지?*** : google 신경망 번역기에서는
인코더 레이어중에 섞여있음. 섞어 쓰면서 long term dependancy를 줄이는 방식으로 
앙상블로 쓰인다. )
- 혹은 입력 시퀀스를 두번 반복하여 네트워크가 더 잘 기억하도록 도움을 주기도 한다. 

### 3. attention
- 논문 : https://arxiv.org/pdf/1409.0473.pdf
- 참고 : http://docs.likejazz.com/attention/
- attention은 모델로 하여금 "중요한 부분만 집중"하도록 만드는 것이 핵심이다. 
- 디코더가 출력을 생성할 때 각 단계별로 입력 시퀀스의 각기 다른 부분에 집중할 수 있도록 만든다. 
- 즉 encoder에서 벡터로 변환할 때 정보 소실이 일어날 수 있는 부분을 조정하는 형태이다. 
- 즉 하나의 고정된 컨텍스트 벡터로 인코딩 하는 대신에 출력의 각 단계별 컨텍스트 벡터를 생성하는 방법을 학습니다. 
- 이는 모델이 입력 시퀀스와 지금까지 생성한 겨로가를 통해 무엇에 집중할지를 학습하는 방식이다. 
![image](https://user-images.githubusercontent.com/49298791/87002226-1a8f4780-c1f4-11ea-9cf9-42f880cefcfe.png)
- 위 그림에서 중요한 점은 출력단어 y_t가 마지막 상태 뿐 아니라 입력 상태의 모든 조합을 참고하고 있다는 점이고 여기서 a는 각 출력이 어떤 입력을 
더 많이 참고하는지에 대한 가중치를 의미한다. 
- 즉 encoder seqeunce에서 contexts로 넘겨줄 때 which encoded charaters to weght high를 보고자 하는 형태이다. 

### 4. transformer
![image](https://user-images.githubusercontent.com/49298791/87006942-6d6cfd00-c1fc-11ea-8e0e-3dba29238d13.png)
- seq2seq의 구조인 encoder, decoder을 따르면서도, attention만으로 구현한 모델이다. 
- 이 모델은 RNN을 사용하지 않고 어텐션만을 사용하여 인코더-디코더 구조를 만들어 학습 속도가 매우 빠르다는 장점이 있다. 
- 앞에선 어텐션을 단순히 RNN을 보정하는 용도로 사용했으나, 보정을 위한 용도가 아닌, 아예 어텐션으로 인코더와 디코더를 만들어보고자함. 
- transformer는 RNN을 사용하진 않지만 기존의 seq2seq처럼 인코더에서 입력 시퀀스를 받고, 디코더에서 출력 시퀀스를 출력하는 인코더-디코더
구조를 유지하고 있습니다. 다만 다른점은 인코더와 디코더의 단위가 N개가 존재한다는 점입니다. 
- RNN이 자연어처리에서 유용했던 점은 단어의 위치에 따라 단어를 순차적으로 입력받아 처리하는 RNN의 특성에 의해 단어의 위치정보를 가질 수 있었다는
점에 있었습니다. 하지만 transformer는 입력을 순차적으로 받는 형식이 아니므로, 단어의 위치 정보를 다른 방식으로 알려줄 필요가 있다.
- 이렇게 단어의 위치 정보를 알려주기 위해 embedding vector에 위치 정보들을 더하여 입력으로 사용하는 방식을 `positional encoding`이라고 한다.
- 어텐션의 종류는 ***(1) encoder self-attention (2) masked decoder self-attention (3) encoder-decoder attention***이 있다. 
 ![image](https://user-images.githubusercontent.com/49298791/87004009-abb3ed80-c1f7-11ea-8930-76d0e48e8e69.png)

#### (1) 인코더
- 해당논문에서는 총 6개의 인코더 층을 사용한다. 
- 인코더를 하나의 층이라는 개념으로 생각하면 하나의 인코더 층에는 2개의 서브 층이 존재한다. 
- 먼저 멀티 헤드 셀프 어텐션은 어텐션을 병렬적으로 사용했다는 의미이고, FFNN은 일반적인 feedfoward의 모양이다. 

***1) 인코더의 멀티-헤드 어텐션***

- 어텐션함수는 주어진 Query에 대해서 모든 key와의 유사도를 각각 구하여 이 유사도를 가중치로 하여 키와 매핑되어 있는 
각각의 value에 반영해준다. 그리고 유사도가 반영된 값을 모두 가중합하여 리턴한다. 
- 예를들어 input 값으로 <The animal didn't cross the street because it was too tired.>에서 it이 의미하는 바를 문장 내 단어끼리의 유사도를 구함으로서 
유사도 높은 단어를 골라냄으로서 알아낸다는 의미이다.
- 셀프 어텐션이 일어나는 과정은 사실 각 단어 벡터들로부터 Q벡터, K벡터, V벡터를 얻는 작업을 거쳐 이 벡터들을 초기 입력인 d_model의 차원을 가지는
단어 벡터들보다 더 작은 차원을 가지는데, 논문에서는 d_model=512차원을 가졌던 벡터들을 64의 차원을 가지는 Q, K, V벡터로 변환했다. 
- 여기서 구한 벡터를 이용하여 attention score을 구하고 softmax함수를 거쳐 attention value를 얻는다. 
![image](https://user-images.githubusercontent.com/49298791/87007188-d3f21b00-c1fc-11ea-8dd1-27257550da8e.png)
- 기본적으로 한번의 어텐션을 하는 것보다 여러번의 어텐션을 병렬로 처리하는 것이 더 효과적이므로 d_model을 num-head개로 나누어
d_model / num_heads의 차원을 가지는 Q,K,V에 대하여 num-heads개의 병렬 어텐션을 수행한다. 

***2) position -wise FFNN***

- multi-head self attention의 결과로 나온 각 값을 FFNN을 정하여 output값을 얻는 작업을 말한다. 

***3) residual connection & layer normalization***

- 잔차연결 참고논문 : https://arxiv.org/pdf/1607.06450.pdf
- 층 정규화 참고논문 : https://arxiv.org/pdf/1607.06450.pdf

#### (2) 디코더
- 인코더와 거의 비슷하나, self-attention시 masked-multi head attention을 사용한다는 점이 다르다. 
-masked를 사용하는 이유는 self attention시 자신의 time step이후에 word는 가려 self-attention되는 것을 막는 역할을 한다. 
- 마지막으로 encoder의 K와 V, decoder의 Q를 서로 attention시켜 위의 seq2seq모델과 종일하게 encoder과 decoder 사이의 관계를 attention시킵니다 
 
### 5.  BERT(이외에 Open AI, ELMo보다 좀더 발전된 형태의 모델이다)
- 논문 : https://arxiv.org/pdf/1810.04805v2.pdf
- 참고 블로그 : https://mino-park7.github.io/nlp/2018/12/12/bert-%EB%85%BC%EB%AC%B8%EC%A0%95%EB%A6%AC/?fbclid=IwAR3S-8iLWEVG6FGUVxoYdwQyA-zG0GpOUzVEsFBd0ARFg4eFXqCyGLznu7w
- 해당 논문을 정리한 내용은 BERT.pdf로 저장함. 
- 대용량의 unlabled로 모델을 미리 학습시킨 후, 특정 task를 가진 labled data로 
transfer learning을 하는 모델
- OpenAI GPT나 ELMo의 경우는 대용량 unlabled corpus를 통해 language model을 
학습하고, 이를 토대로 뒤쪽에 특정 task를 처리하는 network를 붙이는 비슷한 방식을 사용하나,
이 방식들은 shallow bidirectional 혹은 unidirectinal하므로 부족한 부분이 있다. 
- BERT의 경우는 특정 task를 처리하기 위해 새로운 network를 붙일 필요 없이
BERT모델 자체의 fine-tuning을 통해 해당 task의 state-of-the-art를 달성한다. 
- feature based approach : 특정 task를 수행하는 network에 pre-trained language representation을 
추가적인 feature로 제공하여 두개의 network를 붙이는 방식이다. (ELMo)
- fine-tuning approach : task-specific한 parameter을 최대한 줄이고, pre-trained된 
parameter들을 downstream task학습을 통해 조금 바꿔주는 방식이다. (OpenAI GPT)
- BERT pre-training의 새로운 방법론은 아래의 2가지이다. 
(1) Masked Language Model(MLM) : input에 무작위하게 몇개의 token을 mask시키고 이를
transformer구조에 넣어 주변 단어의 context만을 보고 mask된 단어를 예측한다. 
BERT에서는 input전체와 mask된 token을 한번에 transformer encoder에 넣고 원래 token
을 예측하므로 deep bidirectional 이 구현된 형태라고 볼 수가 있다. 
(2) next sentence prediction : 두 문장을 pre-training시에 같이 넣어 두 문장이 이어지는 문장인지
아닌지를 맞추는 것이다. 
 


