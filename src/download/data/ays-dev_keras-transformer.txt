### Requirements

```
$ cat requirements.txt && pip install -r requirements.txt
```

```
tensorflow==2.3.0
numpy==1.18.5
nltk==3.5
```

```
$ tensorboard --version
> 2.4.1

$ pip --version
> pip 21.0.1 from /Users/ays-dev/opt/anaconda3/envs/tf/lib/python3.7/site-packages/pip (python 3.7)

$ python --version
> Python 3.7.9
```


### Train
``python train.py``

```python
import tensorflow as tf
import numpy as np

from dataset import get_dataset, prepare_dataset
from model import get_model


dataset = get_dataset("fr-en")

print("Dataset loaded. Length:", len(dataset), "lines")

train_dataset = dataset[0:100000]

print("Train data loaded. Length:", len(train_dataset), "lines")

(encoder_input,
decoder_input,
decoder_output,
encoder_vocab,
decoder_vocab,
encoder_inverted_vocab,
decoder_inverted_vocab) = prepare_dataset(
  train_dataset,
  shuffle = False,
  lowercase = True,
  max_window_size = 20
)

transformer_model = get_model(
  EMBEDDING_SIZE = 64,
  ENCODER_VOCAB_SIZE = len(encoder_vocab),
  DECODER_VOCAB_SIZE = len(decoder_vocab),
  ENCODER_LAYERS = 2,
  DECODER_LAYERS = 2,
  NUMBER_HEADS = 4,
  DENSE_LAYER_SIZE = 128
)

transformer_model.compile(
  optimizer = "adam",
  loss = [
    "sparse_categorical_crossentropy"
  ],
  metrics = [
    "accuracy"
  ]
)

transformer_model.summary()

x = [np.array(encoder_input), np.array(decoder_input)]
y = np.array(decoder_output)

name = "transformer"
checkpoint_filepath = "./logs/transformer_ep-{epoch:02d}_loss-{loss:.2f}_acc-{accuracy:.2f}.ckpt"

tensorboard_callback = tf.keras.callbacks.TensorBoard(
  log_dir = "logs/{}".format(name)
)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
  filepath = checkpoint_filepath,
  monitor = "val_accuracy",
  mode = "max",
  save_weights_only = True,
  save_best_only = True,
  verbose = True
)

early_stopping_callback = tf.keras.callbacks.EarlyStopping(
  monitor = "val_accuracy",
  mode = "max",
  patience = 2,
  min_delta = 0.001,
  verbose = True
)

transformer_model.fit(
  x,
  y,
  epochs = 15,
  batch_size = 32,
  validation_split = 0.1,
  callbacks=[
    model_checkpoint_callback,
    tensorboard_callback,
    early_stopping_callback
  ]
)
```


##### Output

```
Dataset loaded. Length: 185583 lines
Train data loaded. Length: 100000 lines
Model: "Transformer-Model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
Encoder-Input (InputLayer)      [(None, None)]       0
__________________________________________________________________________________________________
Decoder-Input (InputLayer)      [(None, None)]       0
__________________________________________________________________________________________________
Encoder-Word-Embedding (WordEmb (None, None, 64)     1110016     Encoder-Input[0][0]
__________________________________________________________________________________________________
Decoder-Word-Embedding (WordEmb (None, None, 64)     579456      Decoder-Input[0][0]
__________________________________________________________________________________________________
Encoder-Positional-Embedding (P (None, None, 64)     0           Encoder-Word-Embedding[0][0]
__________________________________________________________________________________________________
Decoder-Positional-Embedding (P (None, None, 64)     0           Decoder-Word-Embedding[0][0]
__________________________________________________________________________________________________
Encoder (Encoder)               (None, None, 64)     66944       Encoder-Positional-Embedding[0][0
__________________________________________________________________________________________________
Decoder (Decoder)               (None, None, 64)     100480      Decoder-Positional-Embedding[0][0
                                                                 Encoder[0][0]
__________________________________________________________________________________________________
Decoder-Output (Dense)          (None, None, 9054)   588510      Decoder[0][0]
==================================================================================================
Total params: 2,445,406
Trainable params: 2,445,406
Non-trainable params: 0
__________________________________________________________________________________________________
Epoch 1/15
2813/2813 [==============================] - ETA: 0s - loss: 1.0506 - accuracy: 0.8396
Epoch 00001: val_accuracy improved from -inf to 0.84285, saving model to ./logs/transformer_ep-01_loss-1.05_acc-0.84.ckpt
2813/2813 [==============================] - 1010s 359ms/step - loss: 1.0506 - accuracy: 0.8396 - val_loss: 0.9235 - val_accuracy: 0.8429
Epoch 2/15
2813/2813 [==============================] - ETA: 0s - loss: 0.4577 - accuracy: 0.9142
Epoch 00002: val_accuracy improved from 0.84285 to 0.87188, saving model to ./logs/transformer_ep-02_loss-0.46_acc-0.91.ckpt
2813/2813 [==============================] - 962s 342ms/step - loss: 0.4577 - accuracy: 0.9142 - val_loss: 0.7430 - val_accuracy: 0.8719
Epoch 3/15
2813/2813 [==============================] - ETA: 0s - loss: 0.3159 - accuracy: 0.9340
Epoch 00003: val_accuracy improved from 0.87188 to 0.88687, saving model to ./logs/transformer_ep-03_loss-0.32_acc-0.93.ckpt
2813/2813 [==============================] - 959s 341ms/step - loss: 0.3159 - accuracy: 0.9340 - val_loss: 0.6626 - val_accuracy: 0.8869
...
Epoch 8/15
2813/2813 [==============================] - ETA: 0s - loss: 0.1545 - accuracy: 0.9600
Epoch 00008: val_accuracy improved from 0.89448 to 0.89624, saving model to ./logs/transformer_ep-08_loss-0.15_acc-0.96.ckpt
2813/2813 [==============================] - 910s 324ms/step - loss: 0.1545 - accuracy: 0.9600 - val_loss: 0.6334 - val_accuracy: 0.8962
Epoch 9/15
2813/2813 [==============================] - ETA: 0s - loss: 0.1441 - accuracy: 0.9621
Epoch 00009: val_accuracy improved from 0.89624 to 0.89654, saving model to ./logs/transformer_ep-09_loss-0.14_acc-0.96.ckpt
2813/2813 [==============================] - 903s 321ms/step - loss: 0.1441 - accuracy: 0.9621 - val_loss: 0.6387 - val_accuracy: 0.8965
Epoch 10/15
2813/2813 [==============================] - ETA: 0s - loss: 0.1346 - accuracy: 0.9640
Epoch 00010: val_accuracy did not improve from 0.89654
2813/2813 [==============================] - 913s 324ms/step - loss: 0.1346 - accuracy: 0.9640 - val_loss: 0.6730 - val_accuracy: 0.8933
Epoch 00010: early stopping
```


### Predict
``python predict.py``

```python
import tensorflow as tf
import numpy as np

from dataset import get_dataset, prepare_dataset
from model import get_model
from utils.make_translate import make_translate


dataset = get_dataset("fr-en")

print("Dataset loaded. Length:", len(dataset), "lines")

train_dataset = dataset[0:100000]

print("Train data loaded. Length:", len(train_dataset), "lines")

(encoder_input,
decoder_input,
decoder_output,
encoder_vocab,
decoder_vocab,
encoder_inverted_vocab,
decoder_inverted_vocab) = prepare_dataset(
  train_dataset,
  shuffle = False,
  lowercase = True,
  max_window_size = 20
)

transformer_model = get_model(
  EMBEDDING_SIZE = 64,
  ENCODER_VOCAB_SIZE = len(encoder_vocab),
  DECODER_VOCAB_SIZE = len(decoder_vocab),
  ENCODER_LAYERS = 2,
  DECODER_LAYERS = 2,
  NUMBER_HEADS = 4,
  DENSE_LAYER_SIZE = 128
)

transformer_model.summary()

transformer_model.load_weights('./logs/transformer_ep-10_loss-0.14_acc-0.96.ckpt')

translate = make_translate(transformer_model, encoder_vocab, decoder_vocab, decoder_inverted_vocab)

translate("c'est une belle journée .")
translate("j'aime manger du gâteau .")
translate("c'est une bonne chose .")
translate("il faut faire à manger pour nourrir les gens .")
translate("tom a acheté un nouveau vélo .")
```


##### Output

```
Original: c'est une belle journée .
Traduction: it' s a beautiful day .
Original: j'aime manger du gâteau .
Traduction: i like to eat some cake .
Original: c'est une bonne chose .
Traduction: that' s a good thing .
Original: il faut faire à manger pour nourrir les gens .
Traduction: we have to feed the people .
Original: tom a acheté un nouveau vélo .
Traduction: tom bought a new bicycle .
```


### Fine-tuning
``python params.py``

```python
import tensorflow as tf
import numpy as np

from tensorboard.plugins.hparams import api as hp

from dataset import get_dataset, prepare_dataset
from model import get_model


dataset = get_dataset("fr-en")

train_dataset = dataset[0:150]

(encoder_input,
decoder_input,
decoder_output,
encoder_vocab,
decoder_vocab,
encoder_inverted_vocab,
decoder_inverted_vocab) = prepare_dataset(
  train_dataset,
  shuffle = True,
  lowercase = True,
  max_window_size = 20
)

x_train = [np.array(encoder_input[0:100]), np.array(decoder_input[0:100])]
y_train = np.array(decoder_output[0:100])

x_test = [np.array(encoder_input[100:150]), np.array(decoder_input[100:150])]
y_test = np.array(decoder_output[100:150])

BATCH_SIZE = hp.HParam("batch_num", hp.Discrete([32, 16]))
DENSE_NUM = hp.HParam("dense_num", hp.Discrete([512, 256]))
HEAD_NUM = hp.HParam("head_num", hp.Discrete([8, 4]))
EMBED_NUM = hp.HParam("embed_num", hp.Discrete([512, 256]))
LAYER_NUM = hp.HParam("layer_num", hp.Discrete([6, 4]))

with tf.summary.create_file_writer("logs/hparam_tuning").as_default():
  hp.hparams_config(
    hparams=[LAYER_NUM, HEAD_NUM, EMBED_NUM, DENSE_NUM, BATCH_SIZE],
    metrics=[
      hp.Metric("val_accuracy")
    ],
  )

def train_test_model(hparams):
  transformer_model = get_model(
    EMBEDDING_SIZE = hparams[EMBED_NUM],
    ENCODER_VOCAB_SIZE = len(encoder_vocab),
    DECODER_VOCAB_SIZE = len(decoder_vocab),
    ENCODER_LAYERS = hparams[LAYER_NUM],
    DECODER_LAYERS = hparams[LAYER_NUM],
    NUMBER_HEADS = hparams[HEAD_NUM],
    DENSE_LAYER_SIZE = hparams[DENSE_NUM]
  )

  transformer_model.compile(
    optimizer = "adam",
    loss = ["sparse_categorical_crossentropy"],
    metrics = ["accuracy"]
  )

  transformer_model.fit(x_train, y_train, epochs = 1, batch_size = hparams[BATCH_SIZE])

  _, accuracy = transformer_model.evaluate(x_test, y_test)

  return accuracy

def run(run_dir, hparams):
  with tf.summary.create_file_writer(run_dir).as_default():
    hp.hparams(hparams)
    accuracy = train_test_model(hparams)
    tf.summary.scalar("val_accuracy", accuracy, step = 1)

session_num = 0

for batch_num in BATCH_SIZE.domain.values:
  for dense_num in DENSE_NUM.domain.values:
    for num_heads in HEAD_NUM.domain.values:
      for num_embed in EMBED_NUM.domain.values:
        for num_units in LAYER_NUM.domain.values:
          hparams = {
              BATCH_SIZE: batch_num,
              DENSE_NUM: dense_num,
              HEAD_NUM: num_heads,
              EMBED_NUM: num_embed,
              LAYER_NUM: num_units
          }
          run_name = "run-%d" % session_num

          print("--- Starting trial: %s" % run_name)
          print({ h.name: hparams[h] for h in hparams })

          run("logs/hparam_tuning/" + run_name, hparams)

          session_num += 1
```


##### Output

```
--- Starting trial: run-0
{'batch_num': 16, 'dense_num': 256, 'head_num': 4, 'embed_num': 256, 'layer_num': 4}
7/7 [==============================] - 1s 136ms/step - loss: 2.3796 - accuracy: 0.5590
2/2 [==============================] - 0s 35ms/step - loss: 1.3560 - accuracy: 0.8240
--- Starting trial: run-1
{'batch_num': 16, 'dense_num': 256, 'head_num': 4, 'embed_num': 256, 'layer_num': 6}
7/7 [==============================] - 1s 203ms/step - loss: 2.9393 - accuracy: 0.5625
2/2 [==============================] - 0s 53ms/step - loss: 1.1636 - accuracy: 0.8240
...
--- Starting trial: run-29
{'batch_num': 32, 'dense_num': 512, 'head_num': 8, 'embed_num': 256, 'layer_num': 6}
4/4 [==============================] - 1s 323ms/step - loss: 6.8676 - accuracy: 0.2650
2/2 [==============================] - 0s 74ms/step - loss: 13.1447 - accuracy: 0.0500
--- Starting trial: run-30
{'batch_num': 32, 'dense_num': 512, 'head_num': 8, 'embed_num': 512, 'layer_num': 4}
4/4 [==============================] - 2s 513ms/step - loss: 7.4851 - accuracy: 0.2650
2/2 [==============================] - 0s 111ms/step - loss: 13.9821 - accuracy: 0.0310
--- Starting trial: run-31
{'batch_num': 32, 'dense_num': 512, 'head_num': 8, 'embed_num': 512, 'layer_num': 6}
4/4 [==============================] - 3s 761ms/step - loss: 7.1519 - accuracy: 0.2660
2/2 [==============================] - 0s 162ms/step - loss: 14.0015 - accuracy: 0.0500
```

### Visualize

```
$ ./tensorboard --logdir=./logs
```


### Configs

```
Base (training on CPU) :
  EMBEDDING_SIZE = 64
  ENCODER_VOCAB_SIZE = 10000
  DECODER_VOCAB_SIZE = 10000
  ENCODER_LAYERS = 2
  DECODER_LAYERS = 2
  NUMBER_HEADS = 4
  DENSE_LAYER_SIZE = 128
  MAX_WINDOW_SIZE = 20
  DATASET_SIZE = 100000
  BATCH_SIZE = 32
 -> 2.4 millions params (model size : ~28 Mo)

Big (training on GPU) :
  EMBEDDING_SIZE = 512
  ENCODER_VOCAB_SIZE = 30000
  DECODER_VOCAB_SIZE = 30000
  ENCODER_LAYERS = 6
  DECODER_LAYERS = 6
  NUMBER_HEADS = 8
  DENSE_LAYER_SIZE = 1024
  MAX_WINDOW_SIZE = 65
  DATASET_SIZE = 200000
  BATCH_SIZE = 32
 -> 60 millions params (model size : ~600 Mo)
```

### Credits

<pre>
<b>Coder un Transformer avec Tensorflow et Keras (LIVE)</b>
<a href="https://www.youtube.com/watch?v=mWA-PmxMBDk">https://www.youtube.com/watch?v=mWA-PmxMBDk</a>
<i>Thibault Neveu</i>
<a href="https://colab.research.google.com/drive/1akAsUAddF2-x57BJBA_gF-v4htyiR12y?usp=sharing">https://colab.research.google.com/drive/1akAsUAddF2-x57BJBA_gF-v4htyiR12y?usp=sharing</a>
</pre>

<pre>
<b>[TUTORIAL + CÓDIGO] Machine Translation usando redes TRANSFORMER (Python + Keras)</b>
<a href="https://www.youtube.com/watch?v=p2sTJYoIwj0">https://www.youtube.com/watch?v=p2sTJYoIwj0</a>
<i>codificandobits</i>
<a href="https://github.com/codificandobits/Traductor_con_redes_Transformer/blob/master/machine-translation-transformers.ipynb">https://github.com/codificandobits/Traductor_con_redes_Transformer/blob/master/machine-translation-transformers.ipynb</a>
</pre>

<pre>
<a href="https://github.com/CyberZHG/keras-transformer">https://github.com/CyberZHG/keras-transformer</a>
<i>CyberZHG</i>
</pre>

<pre>
<b>Dataset</b>
<a href="https://www.manythings.org/anki/">https://www.manythings.org/anki/</a>
</pre>

### Ressources

<https://arxiv.org/abs/1706.03762> Attention Is All You Need (official paper)

<https://github.com/Kyubyong/transformer/blob/fb023bb097e08d53baf25b46a9da490beba51a21/tf1.2_legacy/train.py>

<https://github.com/Kyubyong/transformer/blob/master/model.py>

<https://github.com/pemywei/attention-is-all-you-need-tensorflow/blob/master/Transformer/model/nmt.py>

<https://github.com/tensorflow/models/blob/master/official/nlp/transformer/transformer.py>

<https://towardsdatascience.com/attention-is-all-you-need-discovering-the-transformer-paper-73e5ff5e0634>

<http://jalammar.github.io/illustrated-transformer/>

<https://nlp.seas.harvard.edu/2018/04/03/attention.html>

<http://statmt.org/wmt14/translation-task.html>

<https://github.com/lucidrains/feedback-transformer-pytorch/blob/main/feedback_transformer_pytorch/feedback_transformer_pytorch.py>

<https://huggingface.co/blog/encoder-decoder>

<http://vandergoten.ai/2018-09-18-attention-is-all-you-need/> (softmax implementation)

<https://www.prhlt.upv.es/aigaion2/attachments/Transformer.pdf-54651af972703d4b1443fc03a8a9e9a6.pdf>

<https://github.com/lilianweng/transformer-tensorflow/blob/master/transformer.py>

<https://towardsdatascience.com/transformer-neural-network-step-by-step-breakdown-of-the-beast-b3e096dc857f>

<https://charon.me/posts/pytorch/pytorch_seq2seq_6/#training-the-seq2seq-model>

<https://trungtran.io/2019/04/29/create-the-transformer-with-tensorflow-2-0/>
