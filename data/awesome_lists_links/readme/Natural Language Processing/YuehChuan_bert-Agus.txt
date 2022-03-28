# Use Google BERT to label unstructured text data 

This is the main software repository for the bert-Agus.

- For extensive technical documentation, please refer to [BERT](https://github.com/google-research/bert)
and [bert-as-service](https://bert-as-service.readthedocs.io/en/latest/)

paper: https://arxiv.org/pdf/1810.04805.pdf



## Building

pass: ubuntu 16.04.2, 18.04

lenove x220 cpu 

git clone https://github.com/YuehChuan/bert-Agus.git

use python virtual environment venv,

See here: https://hackmd.io/8HDpYImiQcC0XWQk--vfEw?view 


**pip3 dependency**  see requirements.txt

Get in your venv
```bash=
source venv/bin/activate
```

```bash=
pip install bert-serving-server  # server 
pip install bert-serving-client  #client, independent of 'bert-serving-server' 
pip install opencc-python-reimplemented  #繁簡轉換工具
pip instal scipy==1.1.0 
pip install tensorflow==1.14.0    #tensorflow-gpu==1.14.0 (if you have gpu)
```
   
or just 
  
```
pip install -r requirements.txt
```

## How to fly

**Serving bert server**

### a. Download *chinese_L-12_H-768_A-12* 

Here to download: 

BERT-Base, Chinese:
https://github.com/google-research/bert

or
https://drive.google.com/file/d/18OSCv1kSTwDLvkatktSFmZErhYmrFk2Q/view?usp=sharing
   

### b. **U**nzip chinese_L-12_H-768_A-12 under path 

**bert-Agus/model/**

Modify **PATHNAME** in environment.sh

defaul setting:
   
`export PATHNAME="${ROOT}/model/chinese_L-12_H-768_A-12"`
   
if you have 4 gpu, set **num_worker**
   
`-num_worker=4`


**terminal 1**
   
Simply run,
```bash=
cd ~/bert-Agus

source environment.sh
```

**terminal 2**

modify label_data.py

**step1. Set **Labels** catogory,  加入要標註的類別**

for example:

Labels = ["外部放電", "內部放電", "雜訊干擾"]


**step2. 加入要標記的資料們**

for example

```python=
input_comment = [
    "受DS-TIE外部放電影響。",
    "外部放電。",
    "經現場定位雷丘內部放電",
    "內部放電皮卡皮卡。",
    "受雜訊干擾。",
    "雜訊干擾。",
]
```

`python label_data.py`

Notice
===

Now, everyone is happy!(◕ ‿ ◕ )!
