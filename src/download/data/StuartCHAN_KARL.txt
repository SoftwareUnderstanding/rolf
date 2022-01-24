# KARL

As the knowledge graph can be growing in data everyday, the semantic contents and graph structures are not constant. The knowledge graph can be considered as a representation that extracts the semantic knowledge of the natural language, and the semantic representation of natural language is varying. If we want to automate the self-querying of a knowledge graph, it is necessary to let the query learn the representation of knowledge space along with the data updates. Here, we want a knowledge-graph neural model that can query right to the answer though the knowledge graph goes through updtates. Our neural model takes natural language question as input to query over the varying knowledge graph, and give the answer as output. How to represent the knowledge space and how the query learns the varying knowledge space are essential.  

Is there such architecture that can learn the varying knowledge representation space and accordingly do reasoning tasks over the knowledge graph? [**Atkinson-Shiffrin Memory Model**’s][1] [theory][2] gave the inspiration. We take the input natural language questions as sensory memory, then leveraging the power of attention mechanisms to project the question as embedding vector to query over the knowledge representation space, through which the model self-calibates with the knowledge graph's updates. This is one of our steps to build a self-learning dynamic knowledge graph.  

![Atkinson-Shiffrin Memory Model](./imgs/Figure1.png "Atkinson-Shiffrin Memory Model")

## Architecture

![Architecture](./imgs/Figure0.png "Architecture")  
The model architecture mainly has three components: Sensory Memory, the encoder part incorporated with pretrained language model; Short-term Memory, the decoder part for query tensor generation; Long-term Memory, the interaction of the query tensor generation and the reward function in the reinforcement learning of knowledge-graph vector space.

![Workflow](./imgs/Figure4.png "Workflow")  
The workflow goes through: 1) the sequence-to-sequence structure encodes the question
and decode into a query tensor; 2)in the generated query tensor, each element is an integer index for the entity or
relation embedding; 3)then, the embedded query tensor from step 2 is parsed into triples, and all the embeddings
in the triples of the query are reduced into a vector representing what the query is asking.

## Implementation

We use [PyTorch](http://pytorch.org/) and [Fairseq](https://github.com/pytorch/fairseq) to build our model. Our model takes advantages of [BERT](https://arxiv.org/abs/1810.04805)'s efficiency[3], and adopts inspiration from other [paper](https://arxiv.org/abs/2002.06823)'s efforts[4].

### Preprocessing

The training dataset is from [DBNQA](https://figshare.com/articles/Question-NSpM_SPARQL_dataset_EN_/6118505). We use the byte-pair encoding [subword-nmt](https://github.com/rsennrich/subword-nmt) library to tokenize the subwords and rare words units, which helps to deal with the out-of-vocabulary phenomenon.  

```bash
subword-nmt learn-joint-bpe-and-vocab -i .\data.sparql -o ./bertnmt/code.sparql --write-vocabulary ./bertnmt/voc.sparql

subword-nmt apply-bpe -i ./data.en -c ./bertnmt/code.en -o ./bertnmt/data.bpe.en 

subword-nmt apply-bpe -i ./data.sparql -c ./bertnmt/code.sparql -o ./bertnmt/data.bpe.sparql  

python preprocess.py --source-lang en --target-lang sparql  --trainpref $DATAPATH/train --validpref $DATAPATH/valid --testpref $DATAPATH/test   --destdir $DATAPATH/train --validpref $DATAPATH/valid --testpref $DATAPATH/destdir  --joined-dictionary --bert-model-name bert-base-uncased  

```

After the preprocessing, we get the files as follows:

```bash
    dict.en.txt
    dict.sparql.txt
    test.bert.en-sparql.en.bin
    test.bert.en-sparql.en.idx
    test.en-sparql.en.bin
    test.en-sparql.en.idx
    test.en-sparql.sparql.bin
    test.en-sparql.sparql.idx
    train.bert.en-sparql.en.bin
    train.bert.en-sparql.en.idx
    train.en-sparql.en.bin
    train.en-sparql.en.idx
    train.en-sparql.sparql.bin
    train.en-sparql.sparql.idx
    valid.bert.en-sparql.en.bin
    valid.bert.en-sparql.en.idx
    valid.en-sparql.en.bin
    valid.en-sparql.en.idx
    valid.en-sparql.sparql.bin
    valid.en-sparql.sparql.idx
```

### Training

#### Initializing

```bash
python train.py $DATAPATH/destdir  -a transformer_s2_iwslt_de_en --optimizer adam --lr 0.0005 -s en -t sparql --label-smoothing 0.1  --dropout 0.3 --max-tokens 4000 --min-lr 1e-09 --lr-scheduler inverse_sqrt --weight-decay 0.0001  --criterion label_smoothed_cross_entropy --max-update 150000 --warmup-updates 4000 --warmup-init-lr 1e-07  --adam-betas (0.9,0.98) --save-dir checkpoints/bertnmt0_en_sparql_0.5 --share-all-embeddings  --encoder-bert-dropout --encoder-bert-dropout-ratio 0.5  
```

#### Warmup

```bash
python train.py $DATAPATH  -a transformer_s2_iwslt_de_en --optimizer adam --lr 0.0005 -s en -t sparql --label-smoothing 0.1  --dropout 0.3 --max-tokens 4000 --min-lr 1e-09 --lr-scheduler inverse_sqrt --weight-decay 0.0001  --criterion label_smoothed_cross_entropy --max-update 150000 --warmup-updates 4000 --warmup-init-lr 1e-07  --adam-betas '(0.9, 0.98)' --save-dir checkpoints/bertnmt0_en_sparql_0.5 --share-all-embeddings  --warmup-from-nmt --reset-lr-scheduler  --encoder-bert-dropout --encoder-bert-dropout-ratio 0.5  --warmup-nmt-file $checkpoint_last_file  
```

### Evaluation

To generate the outputs:

```bash
python generate.py  $DATAPATH  --path $checkpoint_last_file   --batch-size 128 --beam 5 --remove-bpe  --bert-model-name  bert-base-uncased   --gen-subset train    --results-path $save_dir
```

We evaluated our model against the [QALD][5] benchmarks on the [GERBIL](6) platform.  

## References 

```bash
[1] https://www.sciencedirect.com/science/article/pii/S0079742108604223

[2] https://en.wikipedia.org/wiki/Atkinson%E2%80%93Shiffrin_memory_model

[3] https://arxiv.org/abs/1810.04805

[4] https://arxiv.org/abs/2002.06823

[5] http://qald.aksw.org/

[6] http://gerbil-qa.aksw.org/gerbil/
```

*****

[1]:https://www.sciencedirect.com/science/article/pii/S0079742108604223

[2]:https://en.wikipedia.org/wiki/Atkinson%E2%80%93Shiffrin_memory_model

[3]:https://arxiv.org/abs/1810.04805

[4]:https://arxiv.org/abs/2002.06823

[5]:http://qald.aksw.org/

[6]:http://gerbil-qa.aksw.org/gerbil/
