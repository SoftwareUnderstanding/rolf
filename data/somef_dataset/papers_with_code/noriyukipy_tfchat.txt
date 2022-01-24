# tfDlg

![](https://github.com/colorfulscoop/tfdlg/workflows/unittest/badge.svg)

**tfDlg** is a Python library for transformer-based language models and dialog models with TensorFlow.

:sparkles: Features :sparkles:

* **Simple models:** tfDlg adopts simple and easy-to-understand model implementation to enable users to customize models for their research and interests. You can find the model implementation in [tfdlg/models.py](tfdlg/models.py). You can utilize these models in the usual manner of tf.keras (e.g. you can call compile and build method for them).
* **Useful utilities:** tfDlg provides several useful utilities. For example,
  * [tfdlg.data](tfdlg/data.py) provides dataset builders to input them to your model. They generate tf.data.Dataset object
  * [tfdlg.schedules](tfdlg/schedules.py) provides learning rate schedules to consider warmup steps as well as linear decay.
  * [tfdlg.losses](tfdlg/losses.py) provides loss function which considers padding.
  * [tfdlg.eval](tfdlg/eval.py) provides function to calculate perplexity.
  * [tfdlg.tokenizers](tfdlg/tokenizers.py) provides SentencePiece tokenizer.
  * [tfdlg.generations](tfdlg/generations.py) provides top-k top-p generator .
* **Utilities for dialog modeling:** Useful utilities for dialog modeling are provided under the `tfdlg.dialog` namespace.
  * [tfdlg.dialog.data](tfdlg/dialog/data.py) provides a dataset builder which considers context of the dialog.

## Installation

Prepare your environment with Python >= 3.8, < 3.9 first.

Then run `pip` to install this package from GitHub.

```sh
$ pip install git+https://github.com/colorfulscoop/tfdlg
```

You can run tests with [pytest](https://docs.pytest.org/en/stable/) to make sure your installtion succeeds.

```sh
$ pip install pytest==6.1.1
$ pytest tests/
```

:memo: If you install tfDlg in a container environment, use the corresponded container.

| GPU use | Container | Command example |
| --- | --- | --- |
| Yes | tensorflow/tensorflow:2.4.1-gpu bash | `docker container run --gpus all -v $(pwd):/work -w /work --rm -it tensorflow/tensorflow:2.4.1-gpu bash` |
| No | python:3.8.7-buster | `docker container run -v $(pwd):/work -w /work --rm -it python:3.8.7-buster bash` |

## Usage

tfDlg provides two ways to use in ways of **script-based** and **package-based**.

tfDlg is a Python package to enable you to use all the functionalities from your Python scripts. This usual way to use tfDlg as a Python package is called a package-based usage here.

On the other hand, script-based utilizes the pacakge to provide fundamental scripts for training, evaluation and serving your models.
In this viewpoint, script-based can be considered as examples of how to use tfDlg as a Python package.

Take a look at the script-based usage first.

### Script-based usage

Get scripts from GitHub first. Then install dependencies.

```sh
$ git clone https://github.com/colorfulscoop/tfdlg
$ cd scripts
$ pip install -r requirements.txt
```

#### Prepare corpus

In this example, we will use [WikiText-2](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/) to train and evaluate a language model.
tfDlg provides a simple script to download the corpus from the official web site in the [example/wikitext](example/benchmark-wikitext) directory.
Use the script [examples/benchmark-wikitext/get_wikitext.py](tfdlg/examples/benchmark-wikitext/get_wikitext.py) to download the data first.

```sh
$ python ../examples/benchmark-wikitext/get_wikitext.py 2_raw
```

This command downloads WikiText-2 consisting of raw level tokens. You can find the corpus under the `wikitext-2-raw` directory.

```sh
$ ls wikitext-2-raw
wiki.test.raw  wiki.train.raw wiki.valid.raw
```

#### Train tokenizer

First of all, you need train your tokenizer. Currently, only [SentencePiece](https://github.com/google/sentencepiece) tokenizer is available.

```sh
$ python train_tokenizer.py tokenizer_model wikitext-2-raw/wiki.train.raw --vocab_size=5000
```

:memo: If you train a tokenizer for languages which do not separate words with white spaces, consider to use `--add_dummy_prefix=False` option to avoid adding a dummy white space at the beginnin of a text (detault is `True` to add a white space at the beginning of a text).


#### Train model

Use the `train_model.py` script to train your model.

```sh
$ python train_model.py --train_file wikitext-2-raw/wiki.train.raw --valid_file wikitext-2-raw/wiki.valid.raw --tokenizer_model_dir tokenizer_model --save_model_dir=model --epochs=10 --batch_size=4 --fp16 --memory_growth
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
decoder (Decoder)            multiple                  88857600
=================================================================
Total params: 88,857,600
Trainable params: 88,857,600
Non-trainable params: 0
_________________________________________________________________
Dataset class: <class 'tfdlg.data.BlockDataset'>
Calculating num_steps
Num steps per epoch: 805
Epoch 1/10
805/805 [==============================] - 373s 310ms/step - loss: 8.1276 - val_loss: 5.0362
Epoch 2/10
805/805 [==============================] - 360s 310ms/step - loss: 4.9925 - val_loss: 4.7901
Epoch 3/10
805/805 [==============================] - 361s 310ms/step - loss: 4.6732 - val_loss: 4.6482
Epoch 4/10
805/805 [==============================] - 361s 310ms/step - loss: 4.4603 - val_loss: 4.5143
Epoch 5/10
805/805 [==============================] - 361s 310ms/step - loss: 4.2946 - val_loss: 4.4289
Epoch 6/10
805/805 [==============================] - 361s 311ms/step - loss: 4.1440 - val_loss: 4.3577
Epoch 7/10
805/805 [==============================] - 361s 311ms/step - loss: 4.0120 - val_loss: 4.3088
Epoch 8/10
805/805 [==============================] - 362s 311ms/step - loss: 3.9132 - val_loss: 4.2694
Epoch 9/10
805/805 [==============================] - 361s 311ms/step - loss: 3.8153 - val_loss: 4.2437
Epoch 10/10
{'loss': 4.2375584, 'perplexity': 69.238594, 'num_batches': 84, 'num_tokens': 344064}
Validation PPL: 69.238594
```

#### Serve web API

The trained model can be serverd as a web API by using the `serve_webapi.py` script.

```sh
$ python serve_webapi.py --tokenizer_model_dir=tokenizer_model --model_dir=model --host="0.0.0.0" --port="8080"
Dataset class: <class 'tfdlg.dialog.data.DialogDataset'>
INFO:     Started server process [435]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
```

Then the request can be sent to the server.

```sh
$ curl http://localhost:8080/generate -d '{"context": ["Long time a go,"]}' -H "content-type:applicaiton/json"
```

The server is launched by [FastAPI](https://fastapi.tiangolo.com/). Therefore a Swagger document is also available on http://localhost:8080/docs.
You can find more detiail of how to use the API in the docuement.

### Library-based usage

As a basic usage, you first need to import config files and models from the package.

```py
from tfdlg.configs import GPT2SmallConfig
from tfdlg.models import PreLNDecoder

config = GPT2SmallConfig()
model = PreLNDecoder(config)
```

Then you can train here in the usual manner of training Tensorflow Keras models by `fit` method.

After training your model, `save_model` saves the model parameter as well as the model hyper parameters which are specified in `tfdlg.configs.Config` class.

```py
from tfdlg.utils import save_model
save_model("path/to/save/dir", model, config)
```

`load_model` can be used to load your model from the directory where you saved it.

```py
from tfdlg.utils import load_model
model = load_model("path/to/save/dir")
```

Check more details in the scripts which are used for script-based usage.

## Model Description

### tfdlg.models.PostLNDecoder

It is the decoder side implementation of [Vaswani+, 2017] .

Difference from [VasWani+, 2017] is

- Weight is initialized by Grolo's uniform distribution except for layers which uses ReLU. For those which uses the ReLU activation function, He's initialization is used. (The weight initialization method is not mentioned in the paper.)

Usage:

```py
from tfdlg.configs import GPT2SmallConfig
from tfdlg.models import PostLNDecoder

config = GPT2SmallConfig()
model = PostLNDecoder(config)
```

### tfdlg.models.PreLNDecoder

PreLNDecoder replaces Post Layer Normalization architecture of PostLNDecoder with Pre Layer Normalization architecture explained in [Xiong+, 2020].

This architecture is related to GPT-2 introduced in [Radford+, 2019].
The main differences from GPT2 are;

- weight initialization is not uniform distribution

Usage:

```py
from tfdlg.configs import GPT2SmallConfig
from tfdlg.models import PreLNDecoder

config = GPT2SmallConfig()
model = PreLNDecoder(config)
```

## Reference

* [Hendrycks+, 2016] *Gaussian Error Linear Units (GELUs)* by Dan Hendrycks and Kevin Gimpel. (https://arxiv.org/abs/1606.08415)
* [Vaswani+, 2017] *Attention Is All You Need* by Ashish Vaswani et al. (https://arxiv.org/abs/1706.03762v5)
* [Radford+, 2018] *Improving Language Understanding by Generative Pre-Training* by Alec Radford et al. (https://openai.com/blog/language-unsupervised/)
* [Radford+, 2019] *Language Models are Unsupervised Multitask Learners* by Alec Radford et al. (https://openai.com/blog/better-language-models/)
* [Holtzman+, 2019] *The Curious Case of Neural Text Degeneration* by Ari Holtzman et al. (https://arxiv.org/abs/1904.09751)
* [Xiong+, 2020] *On Layer Normalization in the Transformer Architecture* by Ruibin Xiong et al. (https://arxiv.org/abs/2002.04745)
