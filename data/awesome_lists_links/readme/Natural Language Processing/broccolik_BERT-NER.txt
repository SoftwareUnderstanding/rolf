# BERT-NER
对应的博客介绍：[BERT微调进行命名实体识别并将模型保存为pb形式](https://blog.csdn.net/broccoli2/article/details/109894132)


BERT作为编码器，分别使用softmax和CRF作为解码器。数据集为行政处罚领域数据，本人分别训练过BERT+softmax和BERT+CRF模型，对比发现CRF解码效果较好，数据太少，目前未做评价。
模型预测结果最终保存为了json格式，如下：
```
"text": "市中区环境保护局关于对四川德元药业集团有限公司的环境行政处罚决定书（川环法内市区罚字〔2017〕03号）发布时间：2017-03-14【字体：大中小】来源：市环保局【打印本页】【关闭窗口】附件：市中区-四川德元药业集团有限公司处罚决定书",
      "entity": {
         "处罚机关": "市中区环境保护局",
         "被处罚公司": "四川德元药业集团有限公司",
         "处罚文书号": "川环法内市区罚字〔2017〕03号",
         "处罚时间": "2017-03-14"
      }
```

### Folder Description:
```
BERT-NER
|____ bert                          # need git from [here](https://github.com/google-research/bert)
|____ cased_L-12_H-768_A-12	    # need download from [here](https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip)
|____ data		            # train data
|____ middle_data	            # middle data (label id map)
|____ output			    # output (final model, predict results)
|____ BERT_NER.py		    # mian code
|____ conlleval.pl		    # eval code
|____ run_ner.sh    		    # run model and eval result
|____BERT_NER_pb.py   		    # run model and eval result and transfer ckpt to saved model (pb)
|____ner_local_pb.py         #load pb and predict

```


### Usage:
```
bash run_ner.sh
```

### What's in run_ner.sh:
```
训练命令：
默认使用softmax解码，crf=True时是使用CRF解码。
python BERT_NER.py\
    --task_name="NER"  \
    --do_lower_case=True \
    --crf=True \
    --do_train=True   \
    --do_eval=False   \
    --do_predict=True \
    --data_dir=data   \
    --vocab_file=chinese_L-12_H-768_A-12/vocab.txt  \
    --bert_config_file=chinese_L-12_H-768_A-12/bert_config.json \
    --init_checkpoint=chinese_L-12_H-768_A-12/bert_model.ckpt   \
    --max_seq_length=128   \
    --train_batch_size=32   \
    --learning_rate=2e-5   \
    --num_train_epochs=2.0   \
    --output_dir=./output/result_dir
   
预测命令：
预测时要选择与训练时一样的模型，如训练时解码用CRF预测时也用CRF解码。

python BERT_NER.py\
    --task_name="NER"  \
    --do_lower_case=True \
    --crf=True \
    --do_predict=True \
    --data_dir=data   \
    --vocab_file=chinese_L-12_H-768_A-12/vocab.txt  \
    --bert_config_file=chinese_L-12_H-768_A-12/bert_config.json \
    --init_checkpoint=chinese_L-12_H-768_A-12/bert_model.ckpt   \
    --max_seq_length=128   \
    --predict_batch_size=8 \
    --output_dir=./output/result_dir    

perl conlleval.pl -d '\t' < ./output/result_dir/label_test.txt
```

核心代码为开源某个仓库的，忘了是哪个，我修改了部分代码，有知道的可以联系我，我给补上引用~

嗯哼，来个star吧~

### reference:

[1] https://arxiv.org/abs/1810.04805

[2] https://github.com/google-research/bert



