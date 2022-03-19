# TensorFlow-MemN2N
End-To-End Memory Networks

TensorFlow version == 1.4

## paper  
    * #https://arxiv.org/abs/1503.08895

## dataset
    * babi (tasks_1-20_v1-2) 
        * https://research.fb.com/downloads/babi/

## MemN2N.py
    * End to End Memory Network를 구현한 class
        
## babi_process.py
    * babi dataset을 전처리 하는 코드.
    
## training.py
    * 전처리된 babi dataset을 학습하는 코드.
    
## result
Jointly Learning  
Model A: Embedding size = 50, Adjacent, Position Encoding, Temporal Encoding  
Model B: Embedding size = 100, Adjacent, Position Encoding, Temporal Encoding, Relu  

| Task | Model A | Model B |
| ------------- | ------------- | ------------- |
| 1 | 0.998 | 0.999 |
| 2 | 0.828 | 0.756 |
| 3 | 0.7 | 0.788 |
| 4 | 0.991 | 0.998 |
| 5 | 0.935 | 0.983 |
| 6 | 0.983 | 1.0 |
| 7 | 0.888 | 0.951 |
| 8 | 0.917 | 0.966 |
| 9 | 0.989 | 1.0 |
| 10 | 0.977 | 0.999 |
| 11 | 0.999 | 1.0 |
| 12 | 0.973 | 0.999 |
| 13 | 0.999 | 1.0 |
| 14 | 1.0 | 0.996 |
| 15 | 1.0 | 1.0 |
| 16 | 0.449 | 0.46 |
| 17 | 0.614 | 0.602 |
| 18 | 0.92 | 0.917 |
| 19 | 0.244 | 0.184 |
| 20 | 1.0 | 1.0 |
| total | 0.8702 | 0.8799 |
