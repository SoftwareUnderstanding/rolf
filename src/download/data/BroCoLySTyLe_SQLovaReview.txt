---
layout: "post"
title: "SQLova"
date: "2019-03-19 17:00"
category: Paper Review
tag: [NLP, wikiSQL, Natural Language to SQL, 고급]
author: 김병건
---

# **SQLova**

 *"A Comprehensive Exploration on WikiSQL with Table-Aware Word Contextualization"* ([arxiv](https://arxiv.org/pdf/1902.01069.pdf))
- Natural Language to SQL task는 자연어로 된 질문을 SQL로 변경하여 DB에 질의를 통해 질문에대한 정답을 이끌어내는 task 입니다. 
- SQLova는 Natural Language to SQL task 의 대표적인 dataset인 wikiSQL의 leader board에서 기존의 모델들보다 훨씬 더 높은 성능으로 State-of-the-art 를 달성해 주목을 받은 모델입니다. (현재는 2위가 되었네요.)
- Language Representation 인 BERT와 기존연구들에 대한 사전지식이 있어야 좀 더 이해하기 수월한 논문입니다.
- 이해를 돕기 위해 wikiSQL dataset을 먼저 살펴보고 이후에 SQLova에 대해서 살펴보도록 하겠습니다.

## wikiSQL dataset

- Salesforce에서 공개한 WikiSQL dataset은 Natural Language to SQL task의 대표적인 dataset 입니다.
- wikiSQL dataset은 Relational Database의 자연어 인터페이스를 구축하기 위한 목적으로 만들어 졌습니다.

![alt text](https://github.com/BroCoLySTyLe/SQLovaReview/blob/master/images/wikiSQL1.png)

- wikiSQL dataset은 위의 그림과 같이 자연어로된 Question과 Table정보를 주고 Question의 정답을 찾기위한 **SQL**을 생성해 정답을 찾아내는 형식의 데이터셋 입니다. 


![alt text](https://github.com/BroCoLySTyLe/SQLovaReview/blob/master/images/wikiSQL2.png)

- wikiSQL dataset은 자연어 Question인 **NL** , Table의 해더정보인 **TBL** , SQL문의 ground truth인 **SQL(T)** , 자연어 Quesion의 정답인 **ANS(T)** 로 구성되어 있습니다.
  -  **SQL(**P**)** 와 **ANS(**P**)** 는 SQLova 모델의 Predicted SQL과 Predicted SQL문을 이용하여 얻은 정답 값이고 위의 데이터 예시처럼 ground truth가 잘못된 것들이 존재합니다.  
  
- wikiSQL dataset은 **SELECT문** 만을 대상으로 하고 있기 때문에 다음과 같이 **6가지의 component**를 구해내면 되는 문제로 치환됩니다.
  - **`select-column` : select-column은 테이블 헤더들(TBL) 중에서 SELECT 문에 들어갈 column을 정해줍니다.** 
    - SQL(T)에서의 Byes에 해당
  - **`select-aggregation` : select-aggragation은 위의 과정에서 정해진 SELECT column에 적용할 avg , max , min , count 등의 연산기호를 정해줍니다.** 
    - SQL(T)에서의 avg()에 해당
  - **`where-number` : WHERE 절의 조건으로 들어갈 column 갯수를 정해줍니다.** 
    - wikiSQL 에서는 WHERE절에서 and 로만 여러개의 column 조건을 이용한 데이터들만 존재합니다.
  - **`where-column` : WHERE 절의 조건으로 들어갈 column들을 정해줍니다.**
    - SQL(T)에서 AND로 연결된 Against와 Wins에 해당
  - **`where-operator` : WHERE 절에 들어갈 조건의 대소비교, 등호 등을 결정합니다.** 
    - SQL(T)에서 =와 <에 해당함 
  - **`where-value` : 자연어로된 Question(NL)로 부터 WHERE 절에 들어갈 조건의 value 값을 결정합니다.** 
    - SQL(T)에서 1076과 13에 해당

 
 ---
## **SQLova paper**
우선적으로 이 논문의 contribution을 먼저 짚고 넘어가겠습니다.

### **contribution**
- Natural Language to SQL과 같은 structured data를 이용하는 task에서 BERT와 같이 문맥을 반영한 단어 정보를 이용할 수 있는 Natural Language Representation를 적용하고 그 효용성을 입증
- Table-Aware BERT 모델을 이용하여서 wikiSQL의 leader board에서 기존 state-of-the-art 성능 대비 큰 성능향상을 보인 것  

### **Table-Aware BERT**


Table-Aware BERT는 기존의 BERT를 이용하여 자연어로 된 질의와 테이블의 헤더정보들을 효과적으로 인코딩하기 위해 만들어진 모델입니다.
기존 BERT에 대한 개념은 ([BERT paper](https://arxiv.org/pdf/1810.04805.pdf))를 참고하시거나 *TmaxAI BERT post* ([Tmax AI BERT post](https://tmaxai.github.io/post/BERT/))를 참고하시면 쉽게 이해할 것 같습니다.



![alt text](https://github.com/BroCoLySTyLe/SQLovaReview/blob/master/images/tableawareBERT.png)

위의 그림에서 보시는것 처럼 Table-Aware BERT는 하늘색으로 표시된 CLS 토큰과 빨강색으로 표시된 자연어 Question 그리고 초록색으로 표시된 테이블의 헤더들 이것들을 각각 구분하기위해 회색으로 표시된 SEP 토큰을 인풋으로 갖게 됩니다.

인풋은 기존의 BERT 처럼 워드의 임베딩값과 position embedding , segment embedding 을 더한 vector값을 인풋으로 가지게 됩니다. 
그렇게 되면 Table-Aware BERT를 통해 각각 토큰의 Hidden Vector값이 나오게 됩니다. 이 word contextualization이 반영된 Hidden Vector 값을 이용하여 뒤에 나올 3가지 model scheme을 가지고 Natural Language to SQL task의 성는을 크게 높인 것이 핵심입니다.

---
#### **3 model scheme**

![alt text](https://github.com/BroCoLySTyLe/SQLovaReview/blob/master/images/3model_scheme.png)

논문에서 설명하는 3가지 모델 scheme에 대해서 살펴보겠습니다. 우선 논문에서는 3가지 모델 scheme중 이 논문에서 중점적으로 다루고 있는 shallow layer에 대해서 우선 살펴보겠습니다. (SQLova는 성능이 가장 잘 나온 C scheme입니다.)


---
#### **Shallow Layer**

![alt text](https://github.com/BroCoLySTyLe/SQLovaReview/blob/master/images/shallow.png)

shallow layer는 어떠한 trainable parameter도 가지고 있지 않은 간단한 구조로 Table-Aware BERT 를 fine-tuning 하기위한 loss function으로만 구성된 layer입니다.

기존의 BERT도 contextualized language representation(BERT)의 우수성을 증명하기 위해 여러 NLP task들을 풀때 새로운 parameter 를 가지는 layer를 추가하기보다는 해당 NLP task를 풀기위한 loss function만 가지고도 좋은 성능을 낸다는 것을 보였고, 이 논문에서도 역시 Natural Language to SQL task에서의 Table-Aware BERT의 우수성을 보이기 위해 Table-Aware BERT를 fine-tuning을 하기위해 loss function으로 구성된 최소의 layer를 구성한 것이 shallow layer 입니다.


![alt text](https://github.com/BroCoLySTyLe/SQLovaReview/blob/master/images/formula1.PNG)

식 (1)을 살펴보면 SELECT 문에 들어갈 column을 정하기 위한 식 입니다. 위의 shallow-layer 그림에서 초록색에 해당하는 테이블 해더들의 히든백터값의 0번째 인덱스 값들을 softmax 하여 어떤 테이블 해더가 SELECT 문에 들어갈 column 인지 정하게 됩니다.

![alt text](https://github.com/BroCoLySTyLe/SQLovaReview/blob/master/images/formula2.PNG)

식 (2)을 살펴보면 SELECT 문에 들어갈 column에 적용할 aggregation(max, min, count, avg 등)을 정해줍니다. 식(1)에서 정해진 테이블 해더의 히든백터의 1번째 ~ 6번째 인덱스 값이 각각 aggregation(NONE ~ AVG)의 score를 가지게 되고 softmax를 통해 한가지의 aggregation을 정하게 됩니다. (NONE이 activation 될경우 aggregation 없음)

![alt text](https://github.com/BroCoLySTyLe/SQLovaReview/blob/master/images/formula3.PNG)

식 (3)을 살펴보면 CLS 히든백터 값을 가지고 where절 조건이 몇개가 필요한지를 나타내는 뮤값 정해줍니다. 위의 shallow-layer 그림에서 하늘색에 해당하는 CLS 히든백터의 0번째~4번째 인덱스값을 가지고 뮤값을 정해주는데 예를들어 0번째 인덱스가 activation 되면 where절에 조건이 0개가 되고 3번째 인덱스가 activation이 되면 where절의 3개의 조건이 and로 엮이게 됩니다.

![alt text](https://github.com/BroCoLySTyLe/SQLovaReview/blob/master/images/formula4.PNG)

식 (4)을 살펴보면 where절에 들어갈 조건이 될 column을 정해줍니다. 위의 shallow-layer 그림에서 초록색에 해당하는 테이블 해더들의 히든백터값의 7번째 인덱스 값들을 sigmoid를 적용하여 where절에 들어갈 조건이 될 column을 정해줍니다. softmax가 아닌 sigmoid를 적용하는 이유는 where절에 들어갈 조건이되는 column이 0개이거나 2개이상일수도 있기 때문입니다.

![alt text](https://github.com/BroCoLySTyLe/SQLovaReview/blob/master/images/formula5.PNG)

식 (5)를 살펴보면 where절에 들어갈 조건이 될 column의 operator를 정해줍니다. 위의 shallow-layer 그림에서 초록색에 해당하는 테이블 해더들의 히든백터값의 8~10번째 인덱스값이 각각 해당 column에 적용될 operator의 score값이 되고 이것을 softmax를 통해 가장 큰 값을 취해 operator를 정해주게 됩니다. 


![alt text](https://github.com/BroCoLySTyLe/SQLovaReview/blob/master/images/formula67.PNG)

마지막으로 식 (6)과 식 (7)을 살펴보면 자연어로 된 질문으로 부터 where절의 value값들을 찾아주게 됩니다.
식 (6)은 자연어로 된 Question에서 where절에 들어갈 value의 시작(start)이 될 지점을 찾아주고 식(7)은 자연어로 된 Question에서 where절에 들어갈 value의 끝(end)이 될 지점을 찾아줍니다. 

식 (6)은 위에서 찾은 뮤값을 통해 뮤값이 2라면 where절에 들어갈 조건이 2개이므로 value가 2개가 되어야하고 첫번째 where절의 value의 시작을 위의 shallow-layer 그림에서 빨간색에 해당하는 자연어 Quesion의 히든백터들의 1번째 인덱스 값들을 softmax를 취해서 찾게 됩니다. 마찬가지로 두번째 where절의 value의 시작은 자연어 Quesion의 히든백터들의 2번째 인덱스 값들을 softmax를 취해서 찾게됩니다.

식 (6)에서 value의 start 지점을 찾았다면 식 (7)에서는 같은 방법으로 value의 end지점을 찾습니다. 첫번째 where절의 value의 끝(end)을 위의 shallow-layer 그림에서 빨간색에 해당하는 자연어 Quesion의 히든백터들의 101(1+100)번째 인덱스 값들을 softmax를 취해서 찾게 됩니다.

그리하여 자연어로 된 Question에서 첫번째 토큰의 1번째 인덱스의 히든백터값과 세번째 토큰의 101번째 인덱스의 히든백터값이 softmax를 통해 선택되면 첫번째 where절 조건의 value는 자연어 Quesion의 첫번째 토큰부터 세번째 토큰까지가 됩니다.

여기까지가 shallow layer에 대한 설명이 되겠습니다.


---
#### **Decoder-Layer & NL2SQL-Layer**

![alt text](https://github.com/BroCoLySTyLe/SQLovaReview/blob/master/images/decoder.png)

Decoder-Layer는 이전 연구인 SQL-Net([arxiv](https://arxiv.org/pdf/1711.04436.pdf))의 모델구조를 가져온 것 입니다. 

SQL-Net에서는 word input으로 word embedding 기법인 GloVe 를 사용한 반면에 이 논문에서는 Table-Aware BERT를 이용해 히든백터 값을 이용하여 word input을 사용하여 Table-Aware BERT의 우수성을 보이고자 했습니다.

![alt text](https://github.com/BroCoLySTyLe/SQLovaReview/blob/master/images/NL2SQL.png)

NL2SQL-Layer(SQLova)는 Table-Aware BERT의 output(히든백터)값을 이용하여 각각 6가지 component를 구할수 있는 모델을 구성한 모델구조로 Trainable한 parameter가 없는 Shallow-Layer보다 보다 복잡한 문제를 풀 수 있게끔 기대되는 모델로 뒤의 실험에서 좋은 성능을 낸 SQLova 모델 입니다.  



---
#### **Experiment**

이제 Experiment 부분을 간단하게 살펴보고 포스트를 마치겠습니다.



![alt text](https://github.com/BroCoLySTyLe/SQLovaReview/blob/master/images/result1.png)

LF는 SQL이 정확하게 생성 되었는지를 판단하는 정확도 이고 X는 자연어로 된 질문에 대한 정답에 대한 정확도 입니다. 당연히 LF가 정확하게 만들어졌다면 X는 맞게 되므로 LF가 좀 더 어려운 task일것 입니다.
EG는 Excution Guide로 SQL EXcution을 위한 룰을 적용한 것이라고 보시면 됩니다. 예를들어 String값을 가지는 column에 MAX나 AVERAGE와 같은 aggregation이 적용되지 않도록 하는 룰입니다.

Table2를 살펴보면 loss fuction으로만 구성된 Shallow-Layer를 가지고 Table-Aware BERT를 fine-tuning한 모델이 기존의 모델들 보다도 훨씬 좋은 성능을 내는 것을 보여줍니다. EG가 없을때는 Shallow-Layer가 NL2SQL-Layer보다도 좋은 성능을 내어 가장 우수한 성능을 내게 됩니다. 하지만 EG를 적용한 경우 NL2SQL이 가장 좋은 성능을 내었고 이 모델이 SQLova가 됩니다.



![alt text](https://github.com/BroCoLySTyLe/SQLovaReview/blob/master/images/result2.png)

Table3은 SQLova와 Shallow-Layer의 성능을 6가지 componenet별로 비교한 것 입니다.



![alt text](https://github.com/BroCoLySTyLe/SQLovaReview/blob/master/images/result3.PNG)

Table4는 ablation study를 진행한 결과인데, 첫번째 줄은 이 논문의 SQLova의 성능이고 두번째줄은 이 논문에서 사용한 BERT-large(Table-Aware BERT)대신에 BERT-Base를 사용한 성능이고, 세번째줄은 Table-Aware BERT의 fine-tuning을 적용하지 않은모델의 성능이고, 네번째줄은 BERT-large대신에 GloVe를 사용했을때의 결과입니다.

첫번째 줄과 두번째 줄의 결과를 비교하였을때, BERT-large의 성능이 BERT-Base보다 뛰어나지만 BERT-Base도 좋은 결과를 내는것을 보여줍니다.

첫번째 줄과 세번째 줄의 결과를 비교하였을때, BERT의 fine-tuning이 성능에 미치는 영향이 매우 크다는 것을 알 수 있습니다.   

첫번째 줄과 네번째 줄의 결과를 비교하였을때, Contextualized word vector를 구성할수 있는 Table-Aware BERT의 적용이 엄청난 성능향상을 보였다는것을 알 수 있습니다.


---

논문에 나오는 그림과 Table의 출처는 "A Comprehensive Exploration on WikiSQL with Table-Aware Word Contextualization"* ([arxiv](https://arxiv.org/pdf/1902.01069.pdf)) 논문이며 설명을 위해 수정한 그림도 있습니다. 

감사합니다.