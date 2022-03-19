# Hierarchical Character Embeddings

Hierarchical character embeddings exploit recursive structures of Chinese characters to construct better characters' representation.
Recursive structures of Chinese characters are obtained from the characters themselves using a rule-based parser.
The rule-based parser data is from [IDS data repository](https://github.com/cjkvi/cjkvi-ids) which is based on the from [CHISE](http://www.chise.org/) project.
Both Simplified and Traditional Chinese input are supported.

If you use this code, please cite as appropriate:

```
@article{nguyen2019hierarchical,
  title={Hierarchical Character Embeddings: Learning Phonological and Semantic Representations in Languages of Logographic Origin Using Recursive Neural Networks},
  author={Nguyen, Minh and Ngo, Gia H and Chen, Nancy F},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  year={2019}
  publisher={IEEE}
}
```

## Requirements

Python and PyTorch are required for the current codebase.
To setup the required environment

1. Install Anaconda
2. Run `conda env create -f env.yml -n hier_emb`



## Examples:

### Usage

```Python
import torch
from nnblk import HierarchicalEmbedding

char2index = {'白': 0, '山': 1, '名': 2, '風': 3}
emb = HierarchicalEmbedding(num_embeddings=len(char2index), embedding_dim=4, char2index=char2index)
input_ = torch.LongTensor(list(char2index.values()))
print(emb(input_))
```

### Language modeling with Hierarchical Character Embedding

Try out the Chinese language modeling example by running `./example_lm_hier_emb.sh`  
The language model (AWD-LSTM-LM) used in the example is described in these two papers:

+ [Regularizing and Optimizing LSTM Language Models](https://arxiv.org/abs/1708.02182)
+ [An Analysis of Neural Language Modeling at Multiple Scales](https://arxiv.org/abs/1803.08240)

