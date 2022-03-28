# QA with WikiSQL


## Task 설명
- 자연어를 통해 인간과 컴퓨터의 상호작용을 연구하는 분야, Natural Language Interface(NLI)
- 자연언어를 SQL 쿼리문으로 변화시켜주는 연구 분야(NL2SQL)


## 데이터 셋
- WikiSQL 데이터 셋 활용([링크](https://github.com/salesforce/WikiSQL))
- 80,654개 자연어 질문, 24,241개 테이블(paper에 작성된 기준)
- WikiSQL에 있는 train.jsonl, test.jsonl, dev.jsonl에서 추출된 정보들.

*데이터 셋* | *Questions* | *Filtered Questions* | *SQL tables* | *Filtered SQL tables* | 
:---: | :---: | :---: | :---: | :---: |
Train | 56,355개 | 56,226개 | 17,290개 | - |
Dev | 8,421개 | 8,393개 | 2,615개 | - |
Test | 15,878개 | 15,841개 | 5,004개 | - |
Total | 80,654개 | 80,460개 | 24,909개 | 24,241개 |

- 테이블 간 중복 정보

*기준* | *Train_table & Dev_table* | *Dev_table & Test_table* | *Test_table & Train_table* | *Total(Train_table & Dev_table & Test_table)* | 
:---: | :---: | :---: | :---: | :---: |
중복 제거 전 | 19,905개 | 7,619개 | 22,294개 | 24,909개 |
중복 제거 후 | 19,720개 | 7,562개 | 21,868개 | 24,241개 |

## 평가
### 1. Tokenizing 성능평가
사전크기(Voca size)와 문장의 길이(sequence length) 사이에는 트레이드오프(trade-off) 관계가 존재한다.
사전에 들어있는 단어가 많아질수록 문장을 단어로 분해했을 때 길이는 줄어들게 된다.
그러나 사전에 미리 지정된 단어가 많아질수록 미지정 된 단어(Unknown, UNK)도 같이 증가하게 된다.
Subword를 사용해서 사전크기와 UNK를 획기적으로 줄이는 연구(BPE)가 [\[2, 3\]](#Reference)에 나와있다.

*Tokenizing 유형* | *Train Voca* | *Train Sequence Length* | *Dev UNK* | *Test UNK* |
:---: | :---: | :---: | :---: | :---: |
stanford + BPE_0(None) | 55,778 | 23.77 | 5,235 | 10,085 |
stanford + BPE_1000 | 2,325 | 41.79 | 83 | 162 |
stanford + BPE_3000 | 4,312 | 33.39 | 84 | 165 |

### 2. NL2SQL 리더보드
- Execution Accuracy(EA) : 쿼리 실행 결과가 정확한 결과를 반환하는지 여부
- Logical Form Accuary(LFA) : 쿼리문이 정답과 일치하는 여부

*모델* | *Dev LFA* | *Dev EA* | *Test LFA* | *Test EA* |
:---: | :---: | :---: | :---: | :---: |
model1 | 0.0 | 0.0 | 0.0 | 0.0 |
baseline | 0.0 | 0.0 | 0.0 | 0.0 |


## Getting Start 
### Requirement
- python 3
- [WikiSQL](https://github.com/salesforce/WikiSQL)
- [corenlp for python](https://github.com/stanfordnlp/python-stanford-corenlp)

### Folder Structure
Home directory : QA_wikisql
```
nli
 |--- QA_wikisql
 |--- WikiSQL
```

### Download Dataset
salesforce의 WikiSQL 깃 레포지토리로부터 데이터 셋과 평가를 위한 코드를 다운로드 한다.
```shell
git clone https://github.com/salesforce/WikiSQL
cd WikiSQL
pip install -r requirements.txt
tar xvjf data.tar.bz2
```

### EDA
데이터 수, 테이블 수, [\[1\]](#Reference)에 작성된 figure 5의 question lengths, number of columns 그림 확인.
논문에서 명시된 데이터 수와 EDA를 수행해서 얻는 데이터 수에 차이가 있으니 이를 잘 유념해서 분석을 수행해야 한다.
질문 길이(Question lengths)는 띄어쓰기를 기준으로 분해된 단어들로 만들어진 그래프이다.
열의 수(Numbers of columns)는 WikiSQL의 {type}.table.jsonl 중 사용된 테이블들의 header 정보로 만들어진 그래프이다. 
```shell
python wikisqlEDA.py
```

### Tokenizing
Dataset의 question과 table column name을 유형별(train, dev, test)로 모아 stanford parser로 tokenizing 진행.
이후 [\[2\]](#Reference), [\[3\]](#Reference) 알고리즘을 적용하여 강건한 input 제작.
Subword로 분해되지 않길 원하는 단어는 preprocess/bpe_util_data/glossaries.txt 파일에 단어를 등록하면 된다.
```shell
python stanford_parsing.py
python learn_bpe.py
python apply_bpe.py --merges=1000

this is example -> this __is __example -> th@@ is __is __ex@@ ample
```

### Check OOV
Tokenizing 성능 평가에 사용되는 코드.
사전도 여기서 획득한다.
```shell
python check_oov.py --merges=1000
```

### Restoring
BPE와 stanford parser결과를 원래 문장으로 복원
공백제거 -> @@를 빈 공간으로 치환 -> __를 띄어쓰기로 치환
```shell
python restore.py --merges=1000 --use_bpe=True

th@@ is __is __ex@@ ample -> th@@is__is__ex@@ample -> this__is__example -> this is example
```

### Get Result
```shell
python evaluate.py --source_file=data/dev.jsonl --db_file=data/dev.db --pred_file=data/example.pred.dev.jsonl
```

## Reference
- [1] [SEQ2SQL: Generating Structured Queries From Natural Language Using Reinforcement Learning](https://arxiv.org/pdf/1709.00103.pdf), arXiv 2017
- [2] [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/pdf/1508.07909.pdf), ACL 2016 
- [3] [BPE-Dropout: Simple and Effective Subword Regularization](https://arxiv.org/pdf/1910.13267.pdf), arXiv 2019
