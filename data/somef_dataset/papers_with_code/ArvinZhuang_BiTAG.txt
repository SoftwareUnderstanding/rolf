# BiTAG: Bidirectional Title/Abstract Generator

BiTAG is a [T5-based](https://arxiv.org/pdf/1910.10683.pdf) text generator that performs two types of generation: 
1) Generate candidate titles for a given abstract (abs_to_title). 
2) Generate abstracts for a given title (title_to_abs).

The basic idea of BiTAG is similar to [docTTTTTquery](https://github.com/castorini/docTTTTTquery) [1] but trained with [BiQDL](https://github.com/ielab/TILDE) [2] loss function. 

BiTAG is trained on 361349 title-abstract pairs that crawled from Arxiv computer science papers uploaded between 2000-06-01 and 2021-06-01. It uses a [Huggingface](https://huggingface.co/transformers/model_doc/t5.html) t5-large model that trained on 4 Tesla v100 GPUs for 4 epochs.


# Minimal use cases
If you just want to use BiTAG to generate title or abstract for you, then you only need to install [transformers](https://github.com/huggingface/transformers) library by `pip install transformers`.

After you install the library, you can download and run BiTAG with the following code:
```
from transformers import AutoModelForSeq2SeqLM, T5Tokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("ArvinZhuang/BiTAG-t5-large")
tokenizer = T5Tokenizer.from_pretrained("t5-large")

text = "abstract: [your abstract]"  # use 'title:' as the prefix for title_to_abs task.
input_ids = tokenizer.encode(text, return_tensors='pt')

outputs = model.generate(
    input_ids,
    do_sample=True,
    max_length=500,
    top_p=0.9,
    top_k=20,
    temperature=1,
    num_return_sequences=10,
)

print("Output:\n" + 100 * '-')
for i, output in enumerate(outputs):
    print("{}: {}".format(i+1, tokenizer.decode(output, skip_special_tokens=True)))

```
Note: the code above does generate candidate titles for your abstract, if you want to generate abstract for a given title, you need to change the prefix `abstract: ` to `title: `.

I also provided some other generation strategies in `generate.py` script. To play around with it, you can simply change the config in `generation_confg.yaml` file and then run `python3 generate.py`.


Here are some title suggestions made by BiTAG for the [BERT](https://arxiv.org/abs/1810.04805) paper.

```
Output:
----------------------------------------------------------------------------------------------------
1: BERT: Training Bidirectional Transformers for Language Representation
2: BERT: Bidirectional Encoder Representations for Natural Language Processing
3: Bidirectional Encoder Representations from Transformers
4: Deep Bidirectional Language Representations from Transformers
5: Bidirectional Encoder Representations from Transformers for Natural Language Processing
6: BERT: A Pre-Trained Language Representation Model
7: BERT: Bidirectional Encoder Representations for Natural Language Processing
8: Pre-trained Models of Unidirectional Representations for Natural Language Processing Tasks using Bidirectional Transformers
9: BiDirectional Encoder Representations from Transformers for Natural Language Processing and Question Answering
10: BERT: Bidirectional Encoder Representations from Transformers
```

Here is an abstract that BiTAG writes for BERT:


Bidirectional Transformer (BT) is a technique for pre-training a deep neural network to learn Bidirectional Transformer Models (DT-MNTs) for language understanding. 
We develop BERT (Bidirectional Bidirectional Transformer) for a DL-to-L LSTM task, where the training set is a mixture of Bidirectional Transformers (BTs) whose inputs are bidirectional, respectively, to train the deep network. However, BERT is typically trained on high-dimensional BERT and, due to the difference between the two tasks, we find that this pre-training has a significant impact on language understanding performance. Extensive experiments on two real-world tasks show that BERT is both accurate and able to achieve competitive performance and efficiency on both the two benchmarks.

# Train BiTAG
We relay on [transformers](https://github.com/huggingface/transformers) and [pytorch-lightning](https://www.pytorchlightning.ai/) to train BiTAG. 

If you want to train BiTAG by yourself, first install the dependencies required in this repo by `pip install -r requirements.txt`.

### Create the training dataset
We use an open-source Arxiv crawler called `arxivscraper` from this [repo](https://github.com/Mahdisadjadi/arxivscraper) to create our training set. Simply run the following command in the root directory:
```
python3 create_dataset.py --date_from 2000-06-01 --date_util 2021-06-01 --category cs
```
This script will create a data folder and store all the title-abstract pairs during this period.

### Training details
After you create your training set, you can simply run `python3 train.py` to start the training.
By default, it will use t5-large and 4 gpus with batch size of 32 per gpu. A model checkpoint will be saved in `ckpts/` at the end of each epoch.

If you want to train a lighter BiTAG, you can use t5-base/small and change batch size and number of gpu by setting the parameter `--batch_size` and
`--num_gpu`




# References
[1] [From doc2query to docTTTTTquery](https://cs.uwaterloo.ca/~jimmylin/publications/Nogueira_Lin_2019_docTTTTTquery-v2.pdf), Rodrigo Nogueira and Jimmy Lin, 2020

[2] [TILDE: Term Independent Likelihood moDEl for Passage Re-ranking](http://ielab.io/publications/pdfs/arvin2021tilde.pdf), Shengyao Zhuang and Guido Zuccon, 2021
