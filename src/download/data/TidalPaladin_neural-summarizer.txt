# Neural Summarizer

This repository provides an implementation for two modified versions of the
extractive summarization architecture
[BertSum](https://github.com/nlpyang/BertSum). This project was carried out as
a final project for CS6320 (Natural Language Processing).

## Description

The two modified architectures are as follows:
1. A dual-transformer model that utilizes both sentence and token level embeddings.
2. A convolutional model that halves the dimensionality of BERT embeddings.

### Dual-Transformer Model

The aim of this architecture is to capture information from the token-level
BERT embeddings without drastically increasing the compute requirements of the
model. A second token-level transformer pathway is added in parallel to the
sentence-level pathway present in the baseline model. Token embeddings for each
input sentence are summed to produce a single vector, and these vectors are
then given as input to the second transformer pathway. The output of both
transformer pathways are then summed and fed through the standard dense /
sigmoid head.

### Convolutional Model

The aim of this architecture is to reduce the compute requirements of BertSum
without sacrificing significantly in performance. A pointwise convolution layer
is applied to BERT embeddings, reducing the dimensionality of these embeddings
from 768 to 384.  These lower dimension embeddings are then passed through the
standard BertSum transformer.

### Lower Precision Model

An attempt was made to train a lower precision model using float16 weights.
This was attempted both manually and through the use of Nvidia's
[Apex](https://github.com/NVIDIA/apex) library, which provides automatic
mixed-precision.  Unfortunately, both of these efforts were unsuccessful. In
the manual approach, errors arose that were likely the result of insufficient
loss scaling. Multiple scaling factors were tried, however the model would fail
after a small number of steps in every case.

## Usage

### Prerequisites

The following items are required:

1. Python 3
2. PyTorch 1.3.1
3. The Python dependencies in `requirements.txt`
4. The ROUGE Perl script (provided in this repository)
5. CNN / Daily Mail dataset ([link](https://github.com/nlpyang/BertSum) to preprocessed dataset)
6. Sufficient hardware to train a BERT model

In order to use the Docker image, both Docker and Docker Compose are required.

### Running

A docker compose file is provided that will build an image and automate the 
training / testing process. 

Create a `.env` file with the following environment variables:

```sh
SRC_DIR=/path/to/dataset
ARTIFACT_DIR=/path/to/logs/and/checkpoints
```

Docker compose will bind these paths to the appropriate mount points within the container.
The `train.sh` and `test.sh` are wrappers that invoke `docker-compose` with the flags used
for our training / testing setup. These scripts can be modified as needed, or `docker-compose`
can be invoked manually. See the [BertSum](https://github.com/nlpyang/BertSum) repository for
more detailed usage instructions pertaining to multi-GPU configurations.


## Results

Due to time and hardware limitations, both the baseline BertSum model and the
two modified architectures were tested at 8000 steps. Training was carried out
on two GTX 970 GPUs, a setup that proved challenging due to the 4GB of memory
on each GTX 970. The learning rate was reduced from the value used in the
BertSum paper to account for the smaller batch sizes used during training. All
models were evaluated using ROUGE-1/2/L scores. ORACLE ROUGE scores were
obtained by selecting up to 5 sentences from an input document such that the
sum of ROUGE-1 and ROUGE-2 scores is maximized.


ROUGE-F at step 8000:

| Model        | ROUGE-1 | ROUGE-2 | ROUGE-L |
| ------------ |:-------:|:-------:|:-------:|
| BertSum      | 15.72   | 4.37    | 13.34   |
| Dual-xformer | 24.88   | 10.56   | 20.99   |
| Lower dim    | 25.22   | 10.69   | 21.36   |

ROUGE-R at step 8000:

| Model        | ROUGE-1 | ROUGE-2 | ROUGE-L |
| ------------ |:-------:|:-------:|:-------:|
| BertSum      | 21.70   | 6.18    | 18.38   |
| Dual-xformer | 39.05   | 16.73   | 32.92   |
| Lower dim    | 36.99   | 15.98   | 31.27   |

Further work is needed to train these models to convergence and reevaluate
their performance. However, the results above indicate that both the
dual-transformer and convolutional architecture are viable competitors to the
baseline BertSum model.

## Future Work
The following improvements / modifications should be pursued in the future:

1. Reevaluate performance with models trained to convergence on suitable hardware.
2. Apply a normalization strategy to the summed token-level vectors.
3. Try various convolutional architectures for dimensionality reduction.
4. Further explore the use of Nvidia's Apex library for mixed precision models.

## References

* Fine-tune BERT for Extractive Summarization [arxiv](https://arxiv.org/pdf/1903.10318.pdf)
* Implementation for [BertSum](https://github.com/nlpyang/BertSum)
* Nvidia's [Apex](https://github.com/NVIDIA/apex) library
* [BERT](https://arxiv.org/abs/1810.04805)
