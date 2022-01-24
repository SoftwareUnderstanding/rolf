
# TurkishNER-BERT


### Folder Description:
bert and multi_cased_L-12_H-768_A-12 should be extracted this directory.


```
BERT-NER
|____ bert                          # need git from [here](https://github.com/google-research/bert)
|____ multi_cased_L-12_H-768_A-12	    # need download from [here](https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip)
|____ data		            # train data (EnglishBERTdata3Labels, TurkishNERdata3Labels, TurkishNERdata7Labels)
|____ middle_data	            # middle data (label id map)
|____ output			    # output (final model, predict results)
|____ BERT_NER.py		    # mian code
|____ run_ner.sh    		    # run model and eval result

```

### Data Sets:

There are 3 different data sets available. 
```
EnglishBERTdata3Labels      # Labels : I-PER B-PER I-ORG B-ORG I-LOC B-LOC
TurkishNERdata3Labels       # Labels : PERSON LOCATION ORGANIZATION
TurkishNERdata7Labels       # Labels : PERSON LOCATION ORGANIZATION DATE TIME PERCENT MONEY
```

### Model:
The model multi_cased_L-12_H-768_A-12 contains 104 different languages where Turkish is in. 

### Usage:
```
bash run_ner.sh
```


### What's in run_ner.sh:
```
python BERT_NER.py\
    --task_name="NER"  \
    --do_train=True   \
    --do_eval=True   \
    --do_predict=True \
    --data_dir=TurkishNERdata7Labels   \
    --vocab_file=multi_cased_L-12_H-768_A-12/vocab.txt  \
    --bert_config_file=multi_cased_L-12_H-768_A-12/bert_config.json \
    --init_checkpoint=multi_cased_L-12_H-768_A-12/bert_model.ckpt   \
    --max_seq_length=128   \
    --train_batch_size=32   \
    --learning_rate=2e-5   \
    --num_train_epochs=3.0   \
    --output_dir=./output/result_dir


javac CRF_Eval.java
java CRF_Eval
```

### How is the Model trained?

Update bert/optimization.py 
```
tvars = tf.trainable_variables()
tvars = [v for v in tvars if 'bert' not in v.name]   ## update the condition 
```

### reference:

[1] https://arxiv.org/abs/1810.04805

[2] https://github.com/google-research/bert



