# Seq2Seq-PyTorch
 Implementation of Seq2Seq(attetion, beamSearch...) with PyTorch 

# 项目结构
---

```
.  
├── README.md  
├── checkpoints            #保存已训练的模型的参数  
├── data  
│   ├── __init__.py  
│   ├── Dataset.py         #dataloader  
│   ├── data_utils.py      #数据预处理相关操作  
│   ├── test.txt           #测试数据  
│   ├── test_ids.txt      
│   ├── train.txt          #训练数据  
│   ├── train_ids.txt   
│   ├── valid.txt          #验证数据  
│   ├── valid_ids.txt  
│   └── vocab              #词典  
├── main.py         
├── models  
│   ├── __init__.py  
│   ├── Attention.py  
│   ├── Decoder.py  
│   ├── Encoder.py  
│   └──  Seq2Seq.py  
├── requirements.txt  
├── results  
└── utils  
    ├── __init__.py  
    ├── Recorder.py        #记录训练过程  
    ├── beamSearch.py      #集束搜索  
    └── greadySearch.py    #贪婪搜索  
```

# 实现
---
## 数据集
DailyDialog（多轮对话，日常生活话题）
* 规模：Total Dialogues	13,118，Average Speaker Turns Per Dialogue	7.9
* Ref：http://yanran.li/dailydialog

## Seq2Seq
* Encoder: GRU
* Decoder: GRU

## Attention
* 键值查询的方式
* Luong等人提出的流程 Ref: https://arxiv.org/abs/1508.04025

## BeamSearch
* 层次遍历实现 Ref:https://blog.csdn.net/u014514939/article/details/95667422?depth_1-utm_source=distribute.pc_relevant.none-task&utm_source=distribute.pc_relevant.none-task

# 使用方法
---
## 安装依赖库
```
$ pip install -r requirements.txt
```

模型训练和评估的函数均位于main.py文件内，通过参数的控制实现,目前可修改的参数如下
```
$ python main.py --help
usage: main.py [-h] [--train_data_root TRAIN_DATA_ROOT]
               [--test_data_root TEST_DATA_ROOT]
               [--valid_data_root VALID_DATA_ROOT]
               [--load_model_path LOAD_MODEL_PATH] [--result_dir RESULT_DIR]
               [--save_model_dir SAVE_MODEL_DIR] [--project PROJECT]
               [--timestamp TIMESTAMP] [--embed_size EMBED_SIZE]
               [--enc_dec_output_size ENC_DEC_OUTPUT_SIZE]
               [--attn_size ATTN_SIZE] [--num_layers NUM_LAYERS]
               [--max_epoch MAX_EPOCH] [--max_len MAX_LEN] [--topk TOPK]
               [--batch_size BATCH_SIZE] [--beam_size BEAM_SIZE]
               [--num_workers NUM_WORKERS] [--lr LR]
               [--scheduler_type SCHEDULER_TYPE]
               [--exponential_lr_decay EXPONENTIAL_LR_DECAY]
               [--step_size STEP_SIZE] [--step_lr_decay STEP_LR_DECAY]
               [--teacher_forcing_ratio TEACHER_FORCING_RATIO]
               [--max_gradient_norm MAX_GRADIENT_NORM] [--use_gpu]
               [--bidirectional] [--seed SEED] [--log_interval LOG_INTERVAL]
               [--test]

```

## 训练

```
$ python main.py [--options]
```
## 评估
```
$ python main.py --test --load_model_path=dir [--options]
```


