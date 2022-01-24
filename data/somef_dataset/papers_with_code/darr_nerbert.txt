# bert_ner

#### 介绍
 Use google BERT to do CoNLL-2003 NER

## dataset

CoNLL-2003 

## model folder

About the parameter --bert_model  

You can just input model name like: bert-base-cased  
It will call pytorch-pretrained-bert to down the bert-base-cased model files.  
And this will work, besides you need to wait a little longer time.  
So if you network is unstable, you may fail.  

At this time, you need to download the pretrained models manually.  
Since this project is based on pytorch, so you need the pytorch version of the BERT.  

pytorch-pretrained-bert has changed name to pytorch-transformers. here is the github:  
[pytorch-transformers](https://github.com/huggingface/pytorch-transformers)

Ok, The Subject create the local model folder  
a model folder is like this:  
```shell
bert-base-cased
    python_model.bin
    bert_config.json
    vocab.txt
```

When you download the model tar.gz file, untar it then you can get python_model.bin and bert_cofnig.json.  
For vocab.txt, you need download the file like bert-base-cased-vocab.txt, and change its name to vocab.txt  


## how to run?

```shell
bash run.sh
```

This command will create the environment that needed by the models.  
This project is created on the purposes of easy-on-run.  
If you want to know the details about the models, just read code.  

It takes about 10 minutes for one epoch, 50 minutes for five epoch.
The results report by BERT paper

```shell
https://arxiv.org/pdf/1810.04805.pdf
```

|System         | Dev F1|Test F1|
|---------------|-------|-------|
|ELMo+BiLSTM+CRF|95.7   |92.2   |
|CVT+Multi      |-      |92.6   |
|BERT_BASE      |96.4   |92.4   |
|BERT_LARGE     |96.6   |92.8   |

The result this project get:
bert-base-cased

```shell
             precision    recall  f1-score   support

        LOC     0.9271    0.9305    0.9288      1668
        ORG     0.8754    0.9055    0.8902      1661
        PER     0.9663    0.9573    0.9618      1617
       MISC     0.7803    0.8348    0.8066       702

avg / total     0.9049    0.9189    0.9117      5648
```
my device has a 1080Ti, if you use bert-large may have a memory problem, and I do not try.

# Inference

```python
from bert import Ner

model = Ner("out_!x/")

output = model.predict("Steve went to Paris")

print(output)
'''
    [
        {
            "confidence": 0.9981840252876282,
            "tag": "B-PER",
            "word": "Steve"
        },
        {
            "confidence": 0.9998939037322998,
            "tag": "O",
            "word": "went"
        },
        {
            "confidence": 0.999891996383667,
            "tag": "O",
            "word": "to"
        },
        {
            "confidence": 0.9991968274116516,
            "tag": "B-LOC",
            "word": "Paris"
        }
    ]
'''
```

# Deploy REST-API
BERT NER model deployed as rest api
In the run.sh file

```bash
python api.py
```
API will be live at `0.0.0.0:8000` endpoint `predict`
#### cURL request
` curl -X POST http://0.0.0.0:8000/predict -H 'Content-Type: application/json' -d '{ "text": "Steve went to Paris" }'`

Output
```json
{
    "result": [
        {
            "confidence": 0.9981840252876282,
            "tag": "B-PER",
            "word": "Steve"
        },
        {
            "confidence": 0.9998939037322998,
            "tag": "O",
            "word": "went"
        },
        {
            "confidence": 0.999891996383667,
            "tag": "O",
            "word": "to"
        },
        {
            "confidence": 0.9991968274116516,
            "tag": "B-LOC",
            "word": "Paris"
        }
    ]
}
```
