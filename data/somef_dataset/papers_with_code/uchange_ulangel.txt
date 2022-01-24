# ulangel

## Background
Ulangel is a python library for NLP text classification. The idea comes from the article of Jeremy Howard et al. "Universal Language Model Fine-tuning for Text Classification" https://arxiv.org/pdf/1801.06146.pdf. The original codes are from the fastai library (https://github.com/fastai/course-v3). We use its NLP part as a source of reference and modify some codes to adapt to our use case. The name `ulangel` comes from `u`niversal `lang`uage mod`el`. It also means the fruit of the research department of company U (a french parisian startup), so called `l'ange de U` (the angel of U). U aimes to describe the ecosystem established by corporates as well as startups by our product `Motherbase`. In Motherbase, we have a large quantity of texts concerning company descriptions, communications where we apply this library `ulangel` to do the Natural Language Processing.

This is a LSTM based neural network. To classify the text, we train at first a language model and then fine-tune it into a classifier. That means the whole system is composed of two parts: language model and the text classifier.
`The language model` is trained to predict the next word based on the input text. Its structure is shown below: ![Language model strucutre](doc/language_model_diagram.jpg)
It is supposed to treat only texts, because other features won't help to predict the next word.

`The classifier` is adapted from the language model: it keeps all layers except the decoder and then adds a pooling layer and a full connected linear neural network in order to classify.
Different from the language model input, there are two kinds of inputs that this library is able to deal with for the text classification:
* `Text only mode`: This input mode means that the input consists of only integers of the text, for exemple: [45, 30, ..., 183]. The classifier structure is shown in the figure below: ![Classifier text only mode](doc/classifier_text_only.jpg)
* `Text plus mode`: This input mode means that the input consists of not only integers of the text, but also other features for the classification problem, for exemple: [[45, 30, ..., 183], True, 2, 2019, ...]. The list [16, 8, 9, 261, ...] is integers of the text as in the `text only mode`. `True` can be a boolean to tell if this text contains a country name, `2` can be the number of presence of the country name, `2019` can be the publication year of this text. You can also add as many features as you want. All features after the integer list can be no matter what you want, as long as they are useful for your classification problem. The classifier structure is shown in the figure below: ![Classifier text plus mode](doc/classifier_text_plus.jpg)


In this library you will find all structures needed to process texts, to pack data, to construct your lstm with dropouts, and some evaluation fuctions, optimizers to train your neural network.

All classes and methodes are seperated in three different parts of this library:
* `ulangel.data`: Preparation of the text classification data, including the text preparation (such as text cleaning) and the data formating (such as creating dataset, creating databunch, padding all texts to have the same length in the same batch, etc.).
* `ulangel.rnn`: Create recurrent neural network structures, such as connection dropouts, activation dropouts, LSTM for language model, encoder, etc.
* `ulangel.utils`: Some tools for training, such as callbacks, optimizers, evaluations functions, etc.


## Install

  ```
  pip install ulangel
  ```

## Usage

### ulangel.data
This part is for the data preparation. There are two main groups of functionalities: `ulangel.data.text_processor` for the text cleaning, and `ulangel.data.data_packer` for the data gathering.

#### ulangel.data.text_processor
In this part, there are methods to clean the text, including:
1. Replace HTML special characters and emoji
2. Replace word repetitions and add `xxwrep` ahead: word word word -> xxwrep 3 word
3. Replace character repetitions and add `xxrep` ahead: cccc -> xxrep 4 c
4. Add spaces around /,@,#,:
5. Remove multiple spaces and keep just one
6. Replace tokens with all letters in capitals by their lower case and add `xxup` ahead: GOOD JOB -> xxup good xxup job
7. Replace tokens with the first letter in capital by their lower caser and add `xxmaj` ahead: We -> xxmaj we

The method `ulangel.data.text_processor.text_proc` calls all methods above to clean the text, tokenize it, and add `xbos` at the beginning and `xfld` at the end of the text. Here is a notebook to show standard text processing steps, including text cleaning and text numeralization: [text_processor](doc/word_embedding_text_processor.ipynb)


#### ulangel.data.data_packer
There are three types of data objects in our training and validation systems. The default input data are numpy.ndarray objects.
* Dataset: Divide(or/and shuffle) input data into batches. Each dataset item is a tuple of x and its corresponding y.
* Dataloader: Here we use the pytorch dataloader, to get dataset item in the way defined by the sampler.
* Databunch: Gathering the training and validation dataloader as one data object and will be given to the learner to train the neural network.


##### Dataset
* `ulangel.data.data_packer.LanguageModelDataset`: For a language model, the input is a bptt-length text and the output is also a text as long as the input with just one word shifted. For a text [w0, w1, w2, w3, w4, ...] (wi is the corresponding integer of a word, in the dictionary of your own text corpus.) If bptt = 4, the input i0 is [w0, w1, w2, w3], then the output o0 will be [w1, w2, w3, w4]. The input and the output are generated from the same text, with the help of the class LanguageModelDataset.
```python
  import numpy as np
  from ulangel.data.data_packer import LanguageModelDataset
  trn_lm = np.load(your_path/'your_file.npy', allow_pickle=True)
  trn_lm_ds = LanguageModelDataset(data=trn_lm, bs=2, bptt=4, shuffle=False)
  # print an item of dataset: (x, y)
  next(iter(trn_lm_ds))
  >>> (tensor([1.1000e+01, 5.0000e+00, 2.0000e+00, 1.0000e+01]),
  tensor([5.0000e+00, 2.0000e+00, 1.0000e+01, 4.0000e+00]))
```

* `ulangel.data.data_packer.TextClassificationDataset`: For a text classifier, its dataset is a little bit different. The input is still the same, but the output is an integer representing the corresponding class label. In this case, we use TextClassificationDataset which inherits the pytorch dataset `torch.utils.data.Dataset`.
```python
  import numpy as np
  from ulangel.data.data_packer import TextClassificationDataset
  trn_ids = np.load(your_path/'your_text_file.npy', allow_pickle=True)
  trn_label = np.load(your_path/'your_label_file.npy', allow_pickle=True)
  trn_clas_ds = TextClassificationDataset(x=trn_ids, y=trn_labels)
  # print an item of dataset: (x, y)
  next(iter(trn_clas_ds))
  >>> (array([   11,     5,     2,    10,     4,     7,     5,     2,     9,
              4]), 2)
```

##### Dataloader
In this library, we use the pytorch dataloader, but with our own sampler. For the language model, batches are generated by concatenating all texts so they all have the same length. We can use directly the dataloader to pack data.
```python
  from torch.utils.data import DataLoader
  trn_lm_dl = DataLoader(trn_lm_ds, batch_size=2)
  # print an item of dataloader: a batch of dataset
  next(iter(trn_lm_dl))
  >>> [tensor([[11.,  5.,  2., 10.],
           [12., 11.,  5.,  2.]]), tensor([[ 5.,  2., 10.,  4.],
           [11.,  5.,  2., 10.]])]
```

However, for the text classification, we can not concatenate texts together, because each text has its own class. It doesn't make sense to mix texts to form equilong texts in the batch. In order to train the neural network in an efficient way and at the same time to keep the randomness, we have two different samplers for the training and the validation data. Additionally, for texts in each batch, they should have the same length, so we use a collate function to pad those short texts.
For the Classification data, we use the dataloader in this way:
```python
  from ulangel.data.data_packer import pad_collate_textonly
  trn_clas_dl = DataLoader(trn_clas_ds, batch_size=2, sampler=trn_sampler, collate_fn=pad_collate_textonly)
  val_clas_dl = DataLoader(val_clas_ds, batch_size=2, sampler=val_sampler, collate_fn=pad_collate_textonly)
```
How to create samplers and collat_fn will be explained below.

###### Sampler
Sampler is an index generator. It returns a list of indexes, which corresponding item is sorted by the attribute key. In this library, TrainingSampler and ValidationSampler inherit the pytorch sampler `torch.utils.data.Sampler`.

* `ulangel.data.data_packer.TrainingSampler`: TrainingSampler is a sampler for the training data. It sorts the data in the way defined by the given key, the longest at the first, the shortest at the end, random in the middle.
```python
  from ulangel.data.data_packer import TrainingSampler
  trn_sampler = TrainingSampler(data_source=trn_clas_ds.x, key=lambda t: len(trn_clas_ds.x[t]), bs=2)
```
In this exemple, the data source is the x of the training dataset (texts), the key is the length of each text.

* `ulangel.data.data_packer.ValidationSampler`: It sorts the data in the way defined by the given key, in an ascending or a descending way. It is different from the TrainingSampler, there is no randomness, the validation texts will be sorted from the longest to the shortest.
```python
  from ulangel.data.data_packer import ValidationSampler
  val_sampler = ValidationSampler(val_clas_ds.x, key=lambda t: len(val_clas_ds.x[t]))
```
In this exemple, the data source is the x of the validation dataset (texts), the key is the length of each text.

###### Collate Function
Collate function can be used to manipulate your input data. In this library, our collate function: `pad_collate` is to pad all texts with padding index pad_idx to have the same length for one whole batch. This pad_collate function is inbuild, we just need to import, so that we can use it in the dataloader. It exists for two different input modes.
* `ulangel.data.data_packer.pad_collate_textonly`:
```python
  from ulangel.data.data_packer import pad_collate_textonly
```

* `ulangel.data.data_packer.pad_collate_textplus`: is the textplus version.
```python
  from ulangel.data.data_packer import pad_collate_textplus
```

##### Databunch
* `ulangel.data.data_packer.DataBunch`: Databunch packs your training dataloader and validation dataloader together into a databunch object, so that your can give it to your learner (which will be explained later in README)
```python
  from ulangel.data.data_packer import DataBunch
  language_model_data = DataBunch(trn_lm_dl, val_lm_dl)
```

### ulangel.rnn
In this part, there are two main blocks to build a neural network: dropouts and some special neural network structures for our transfer learning.

#### ulangel.rnn.dropouts
The pytorch dropout are dropouts to zero out some activations with probability p. In the article of Stephen Merity et al. "Regularizing and Optimizing LSTM Language Models" https://arxiv.org/pdf/1708.02182.pdf they propose to apply dropouts not only on activations but also on connections. Using pytorch dropouts is not enough. Therefore, we create three different dropout classes:

* `ulangel.rnn.dropouts.ActivationDropout`: as its name, this is a dropout to zero out activations in the layer of the neural network.
* `ulangel.rnn.dropouts.ConnectionWeightDropout`: as its name, this is a dropout to zero out connections (weights) between layers.
* `ulangel.rnn.dropouts.EmbeddingDropout`: this is a dropout class to zero out embedding activations.

These three dropout classes will be used in the AWD_LSTM to build the LSTM for the language model training. Of course, you can also import them to build your own neural network.

#### ulangel.rnn.nn_block
In this part, we have some structures to build a language model and a text classifier.

* `ulangel.rnn.nn_block.AWD_LSTM`: is the class to build the language model except the decoder layer. This is the commom part for the language model and the text classifier. It is a LSTM neural network inheriting `torch.nn.Module` proposed by Stephen Merity et al. in the article "Regularizing and Optimizing LSTM Language Models" https://arxiv.org/pdf/1708.02182.pdf. Because we use the pretrained language model wikitext-103 from this article, to finetune our own language model on our corpus, we need to keep the same values for some hyperparameters as wikitext-103: embedding_size = 400, number_of_hidden_activation = 1150.
```python
  from ulangel.rnn.nn_block import AWD_LSTM
  # define hyperparameters
  class LmArg:
      def __init__(self):
          self.number_of_tokens = 10000
          self.embedding_size = 400
          self.pad_token = 1
          self.embedding_dropout = 0.05
          self.number_of_hidden_activation = 1150
          self.number_of_layers = 3
          self.activation_dropout = 0.3
          self.input_activation_dropout = 0.65
          self.embedding_activation_dropout = 0.1
          self.connection_hh_dropout = 0.5
          self.decoder_activation_dropout = 0.4
          self.bptt = 16
  encode_args = LmArg()
  # build the LSTM neural network
  lstm_enc = AWD_LSTM(
      vocab_sz=encode_args.nomber_of_tokens,
      emb_sz=encode_args.embedding_size,
      n_hid=encode_args.number_of_hidden_activation,
      n_layers=encode_args.number_of_layers,
      pad_token=encode_args.pad_token,
      hidden_p=encode_args.activation_dropout,
      input_p=encode_args.input_activation_dropout,
      embed_p=encode_args.embedding_activation_dropout,
      weight_p=encode_args.connection_hh_dropout)
  lstm_enc
  >>> AWD_LSTM(
    (emb): Embedding(10000, 400, padding_idx=1)
    (emb_dp): EmbeddingDropout(
      (emb): Embedding(10000, 400, padding_idx=1)
    )
    (rnns): ModuleList(
      (0): ConnectionWeightDropout(
        (module): LSTM(400, 1150, batch_first=True)
      )
      (1): ConnectionWeightDropout(
        (module): LSTM(1150, 1150, batch_first=True)
      )
      (2): ConnectionWeightDropout(
        (module): LSTM(1150, 400, batch_first=True)
      )
    )
    (input_dp): ActivationDropout()
    (hidden_dps): ModuleList(
      (0): ActivationDropout()
      (1): ActivationDropout()
      (2): ActivationDropout()
    )
  )
```

* `ulangel.rnn.nn_block.LinearDecoder`: This is a decoder inheriting `torch.nn.Module`, the inverse of an encoder, to transfer the last hidden layer (embedding vector) into its corresponding integer representation of the work, so that we can find comprehensive words for human.
```python
  from ulangel.rnn.nn_block import LinearDecoder
  decoder = LinearDecoder(
      encode_args.number_of_tokens,
      encode_args.embedding_size,
      encode_args.decoder_activation_dropout,
      tie_encoder=lstm_enc.emb,
      bias=True
  )
  decoder
  >>>LinearDecoder(
    (output_dp): ActivationDropout()
    (decoder): Linear(in_features=400, out_features=10000, bias=True)
  )
```

* `ulangel.rnn.nn_block.SequentialRNN`: This class inherits the pytorch class `torch.nn.Sequential`, to connect different neural networks, and allows to reset all parameters of substructures with a reset methode (ex: AWD_LSTM)
```python
  from ulangel.rnn.nn_block import SequentialRNN
  language_model = SequentialRNN(lstm_enc, decoder)
  language_model.modules
  >>>
  <bound method Module.modules of SequentialRNN(
    (0): AWD_LSTM(
      (emb): Embedding(10000, 400, padding_idx=1)
      (emb_dp): EmbeddingDropout(
        (emb): Embedding(10000, 400, padding_idx=1)
      )
      (rnns): ModuleList(
        (0): ConnectionWeightDropout(
          (module): LSTM(400, 1150, batch_first=True)
        )
        (1): ConnectionWeightDropout(
          (module): LSTM(1150, 1150, batch_first=True)
        )
        (2): ConnectionWeightDropout(
          (module): LSTM(1150, 400, batch_first=True)
        )
      )
      (input_dp): ActivationDropout()
      (hidden_dps): ModuleList(
        (0): ActivationDropout()
        (1): ActivationDropout()
        (2): ActivationDropout()
      )
    )
    (1): LinearDecoder(
      (output_dp): ActivationDropout()
      (decoder): Linear(in_features=400, out_features=10000, bias=True)
    )
  )>
```

For the classification data, there are two types of input data: `text only mode` and `text plus mode` as mentioned in the Background. Therefore all structures concerning classifier are made in two versions for these two different modes of input data.

* `ulangel.rnn.nn_block.TextOnlySentenceEncoder`: it is a class similar to `ulangel.rnn.nn_block.AWD_LSTM`, but the difference is when the input text length exceeds the value of bptt (we define to train the language model), it divides the text into serval bptt-length sequences at the input and concatenates the results back to one text at the output.
```python
  from ulangel.rnn.nn_block import TextOnlySentenceEncoder
  sent_enc = TextOnlySentenceEncoder(lstm_enc, encode_args.bptt)
  sent_enc
  >>>TextOnlySentenceEncoder(
    (module): AWD_LSTM(
      (emb): Embedding(10000, 400, padding_idx=1)
      (emb_dp): EmbeddingDropout(
        (emb): Embedding(10000, 400, padding_idx=1)
      )
      (rnns): ModuleList(
        (0): ConnectionWeightDropout(
          (module): LSTM(400, 1150, batch_first=True)
        )
        (1): ConnectionWeightDropout(
          (module): LSTM(1150, 1150, batch_first=True)
        )
        (2): ConnectionWeightDropout(
          (module): LSTM(1150, 400, batch_first=True)
        )
      )
      (input_dp): ActivationDropout()
      (hidden_dps): ModuleList(
        (0): ActivationDropout()
        (1): ActivationDropout()
        (2): ActivationDropout()
      )
    )
  )
```

* `ulangel.rnn.nn_block.TextPlusSentenceEncoder`: is the text plus version of the `SentenceEncoder`
```python
  from ulangel.rnn.nn_block import TextPlusSentenceEncoder
  sent_enc = TextPlusSentenceEncoder(lstm_enc, encode_args.bptt)
  sent_enc
  >>>TextPlusSentenceEncoder(
    (module): AWD_LSTM(
      (emb): Embedding(10000, 400, padding_idx=1)
      (emb_dp): EmbeddingDropout(
        (emb): Embedding(10000, 400, padding_idx=1)
      )
      (rnns): ModuleList(
        (0): ConnectionWeightDropout(
          (module): LSTM(400, 1150, batch_first=True)
        )
        (1): ConnectionWeightDropout(
          (module): LSTM(1150, 1150, batch_first=True)
        )
        (2): ConnectionWeightDropout(
          (module): LSTM(1150, 400, batch_first=True)
        )
      )
      (input_dp): ActivationDropout()
      (hidden_dps): ModuleList(
        (0): ActivationDropout()
        (1): ActivationDropout()
        (2): ActivationDropout()
      )
    )
  )
```

* `ulangel.rnn.nn_block.TextOnlyPoolingLinearClassifier`: different from the language model, we don't need the decoder to read the output. We want to classify the input text. So at the output of the `ulangel.rnn.nn_block.AWD_LSTM`, we do some pooling to pick the last sequence of the LSTM's output, the max pooling of the LSTM's output, the average pooling of the LSTM's output. We concatenate these three sequences, as input of a linear full connected neural net classifier. The last layer's number of activations should be the same as the number of classes in your classification problem.
```python
  from ulangel.rnn.nn_block import TextOnlyPoolingLinearClassifier
  pool_clas = TextOnlyPoolingLinearClassifier(
      layers=[3*encode_args.emsize, 100, 4], # define the number of activations for each layer
      drops=[0.2, 0.1])
  pool_clas
  >>>TextOnlyPoolingLinearClassifier(
    (layers): Sequential(
      (0): BatchNorm1d(1200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (1): Dropout(p=0.2, inplace=False)
      (2): Linear(in_features=1200, out_features=100, bias=True)
      (3): ReLU(inplace=True)
      (4): BatchNorm1d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (5): Dropout(p=0.1, inplace=False)
      (6): Linear(in_features=100, out_features=4, bias=True)
      (7): ReLU(inplace=True)
    )
  )
```

* `ulangel.rnn.nn_block.TextPlusPoolingLinearClassifier`: is the text plus version. The difference from the text only mode is that text plus mode pooling linear classifier has another group of layers. This supplemental group of layers takes nonverbal features into account.
```python
  from ulangel.rnn.nn_block import TextPlusPoolingLinearClassifier
  pool_clas = TextPlusPoolingLinearClassifier(
      layers1=[3*encode_args.emsize, 100, 4], # structure of text only classification
      drops1=[0.2, 0.1],
      layers2=[9, 4], # structure that you want with other features
      drops2=[0.1])
  pool_clas
  >>>TextPlusPoolingLinearClassifier(
      (layers1): Sequential(
          (0): BatchNorm1d(1200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (1): Dropout(p=0.2, inplace=False)
          (2): Linear(in_features=1200, out_features=100, bias=True)
          (3): ReLU(inplace=True)
          (4): BatchNorm1d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (5): Dropout(p=0.1, inplace=False)
          (6): Linear(in_features=100, out_features=4, bias=True)
          (7): ReLU(inplace=True)
      )
      (layers2): Sequential(
          (0): BatchNorm1d(9, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (1): Dropout(p=0.1, inplace=False)
          (2): Linear(in_features=9, out_features=4, bias=True)
          (3): ReLU(inplace=True)
      )
)
```

To build the complete classifier, we use the `ulangel.rnn.nn_block.SequentialRNN` to connect these two classes, here is an exemple for the text only mode:
```python
  classifier = SequentialRNN(sent_enc, pool_clas)
  classifier
  >>>SequentialRNN(
    (0): TextOnlySentenceEncoder(
      (module): AWD_LSTM(
        (emb): Embedding(10000, 400, padding_idx=1)
        (emb_dp): EmbeddingDropout(
          (emb): Embedding(10000, 400, padding_idx=1)
        )
        (rnns): ModuleList(
          (0): ConnectionWeightDropout(
            (module): LSTM(400, 1150, batch_first=True)
          )
          (1): ConnectionWeightDropout(
            (module): LSTM(1150, 1150, batch_first=True)
          )
          (2): ConnectionWeightDropout(
            (module): LSTM(1150, 400, batch_first=True)
          )
        )
        (input_dp): ActivationDropout()
        (hidden_dps): ModuleList(
          (0): ActivationDropout()
          (1): ActivationDropout()
          (2): ActivationDropout()
        )
      )
    )
    (1): TextOnlyPoolingLinearClassifier(
      (layers): Sequential(
        (0): BatchNorm1d(1200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (1): Dropout(p=0.2, inplace=False)
        (2): Linear(in_features=1200, out_features=100, bias=True)
        (3): ReLU(inplace=True)
        (4): BatchNorm1d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): Dropout(p=0.1, inplace=False)
        (6): Linear(in_features=100, out_features=4, bias=True)
        (7): ReLU(inplace=True)
      )
    )
  )
```
For the text plus mode it will be the same. Just need to be careful that sentence encoder and pooling linear classifier should be in the same mode (an TextOnlySentenceEncoder should not be followed by a TextPlusPoolingLinearClassifier).

### ulangel.utils
In this part, there are some tools for the training of the neural network.

#### Callbacks
Callbacks are triggers during the training. Calling callbacks can make intermediate computation or do the setting.

* `ulangel.utils.callbacks.TrainEvalCallback`: setting if the model is in the training mode or in the validation mode. During the training mode, update the progressing and the number of iteration.

* `ulangel.utils.callbacks.TextOnlyCudaCallback`: putting the model and the variables on cuda.

* `ulangel.utils.callbacks.TextPlusCudaCallback`: is the textplus version to put the model and the variables on cuda.

* `ulangel.utils.callbacks.Recorder`: recording the loss value and the learning rate of every batch, plot the variation of these two values if the methode (recorder.plot_lr() / recorder.plot_loss() / recorder.plot()) is called.

* `ulangel.utils.callbacks.LR_Find`: giving the minimum and the maximum of learning rate and the maximum number of iteration, change linearlly the learning rate (from the minimum value to the maximum value) at every batch. Combine with the Recorder, we can see the evaluation of loss so that we can find an appropriate learning rate for the training. Warning: if there is LR_Find in the callback list, the model is running to go through all learning rates, but not to train the model.

* `ulangel.utils.callbacks.RNNTrainer`: recording the prediction result, raw_output (without applying dropouts) and output (after applying dropouts) after every prediction. If needed, it can also add AR or/and TAR regularization to the loss to avoid overfitting.

* `ulangel.utils.callbacks.ParamScheduler`: allowing to schedule any hyperparameter during the training, such as learning rate, momentum, weight decay, etc. It takes the hyperparameter's name and its schedule function sched_func. Here we use a combined schedule function combine_scheds, combing two different parts of a cosine function, to have a learning rate low at the beginning and at the end, high in the middle.
```python
  from ulangel.utils.callbacks import combine_scheds, ParamScheduler, sched_cos
  lr = 1e-3
  sched_cos1 = sched_cos(start=lr/10, end=lr*2)
  sched_cos2 = sched_cos(start=lr*2, end=lr/100)
  # pcts means the percentages taken by the following functions in scheds. In the exemple below means the sched combines the first 0.3 of sched_cos1 and the last 0.7 of sched_cos2.
  sched = combine_scheds(pcts=[0.3, 0.7], scheds=[sched_cos1, sched_cos2])
```
 The scheduled learing rate defined above looks like this:
 ![scheduled learning rate](doc/learning_rate_scheduler.png)


For the training process, it's up to the user to choose callbacks to make a callback list. Here it's an exemple:
```python
  from ulangel.utils.callbacks import TrainEvalCallback, TextOnlyCudaCallback, Recorder, RNNTrainer
  cbs = [TextOnlyCudaCallback(), TrainEvalCallback(), Recorder(), RNNTrainer(alpha=2., beta=1.), ParamScheduler('lr', sched)]
```


#### Stats
Stats contains all classes and functions to compute statistics of the model's performance. There are two classes and some methods.

* metrics: A metric function takes the outputs of your model, and the target values as inputs, and you can define your own way to evaluate your model's performance by writing your own computation in the function. Functions `ulangel.utils.stats.accuracy_flat` (calculate the accuracy for the language model) and `ulangel.utils.stats.accuracy` (calculate the accuracy for the classifier) are two inbuild metrics that we provide.
Warning: `ulangel.utils.stats.cross_entropy_flat` is not a metric. It is a loss function, but it is similar to the `accuracy_flat` metric, so we put them at the same place.

* `ulangel.utils.stats.AvgStats`: calculate loss and statistics defined by input metrics. This class puts the loss value and other performance statistics defined by metrics together into a list. It also has methods to update and print all these performance statistics when called.

* `ulangel.utils.stats.AvgStatsCallback`: Actually the class `AvgStatsCallback` is also a callback, it uses AvgStats to calculate all performance statistics after every batch, and print these statistics after every epoch.

We can add AvgStatsCallback into the callback list, so that we can know the neural network performs after every epoch.

```python
  from ulangel.utils.stats import AvgStatsCallback, accuracy, accuracy_flat
  # for a language model
  cbs_languagemodel = [TextOnlyCudaCallback(), TrainEvalCallback(), AvgStatsCallback([accuracy_flat]), Recorder(), RNNTrainer(alpha=2., beta=1.), ParamScheduler('lr', sched)]

  # for a classifier
  cbs_classifier = [TextOnlyCudaCallback(), TrainEvalCallback(), AvgStatsCallback([accuracy]), Recorder(), ParamScheduler('lr', sched_clas)]
```


#### Optimizer
* optimizers: `ulangel.utils.optimizer.Optimizer` is a class that decides the way to update all parameters of the model by steppers. `ulangel.utils.optimizer.StatefulOptimizer` is an optimizer with state. It inherits the class `Optimizer` and adds an attribute `state` in order to track the history of updates. As we know, when we use an optimizer with momentum, we need to know the last update value to calculate the current one. In this case, we use `StatefulOptimizer` in this library.

* stepper: functions defining how to update the parameters or the gradient of the parameters. It depends on the current values. In the library we provide several steppers: `ulangel.utils.optimizer.sgd_step` (stochastic gradient descent stepper), `ulangel.utils.optimizer.weight_decay` (weight decay stepper), `ulangel.utils.optimizer.adam_step` (adam stepper). You can also program your own stepper.

* stateupdater: define how to initialize and update state (for exemple, how to update momentum). In the library we provide some inbuild stateupdaters, all of them inherit the class `ulangel.utils.optimizer.StateUpdater`: `ulangel.utils.optimizer.AverageGrad`(momentum created by averaging the gradient), `ulangel.utils.optimizer.AverageSqrGrad`(momentum created by averaging the square of the gradient), `ulangel.utils.optimizer.StepCount`(step increment).

In ulangel, we provide two inbuild optimizer: `ulangel.utils.optimizer.sgd_opt` (stochastic gradient descent optimizer) and `ulangel.utils.optimizer.adam_opt` (adam optimizer). Optimizer is an input of the object of the class leaner. We will show you how to use optimizer in the learner part.

If you want to, you can also write your own stepper, your own stateupdater, to build your own optimizer. Here is an exemple to build an optimizer with momentum.

```python
  from ulangel.utils.optimizer import StatefulOptimizer, StateUpdater, StepCount

  def your_stepper(p, lr, *args, **kwargs):
      p = your_stepper_function(p, lr)
      return p

  def your_stateupdater(Stat):
      def __init__(self):
          your initialization values

      def init_state(self, p):
          return {"your_state_name": your_state_initialization_function(p)}

      def update(self, p, state, *args, **kwargs):
          state["your_state_name"] = your_state_update_function(p, *args)
          return state

  def your_optimizer(xtra_step=None, **kwargs):
      return partial(
          StatefulOptimizer,
          steppers=[your_stepper] + listify(xtra_step),
          stateupdaters=[your_stateupdater()],
          **kwargs
      )
```


#### ulangel.utils.learner
This part includes the class `Learner` and some methods to freeze or unfreeze layers in order to train just a part of or the whole neural network.

##### Learner
* The class `ulangel.utils.learner.Learner`: It is a class that takes the RNN model, data for training, the loss function, the optimizer, the learning rate and callbacks that you need. The method `Learner.fit(epochs=number of epochs that you want to train)` executes all processes in order to train the model. Here is an exemple to build the langage model learner:
```python
  from ulangel.utils.learner import Learner
  language_model_learner = Learner(
      model=language_model,
      data=language_model_data,
      loss_func=cross_entropy_flat,
      opt_func=adam_opt(),
      lr=1e-5,
      cbs=cbs_languagemodel)
  # load the pretrained model
  wgts = torch.load('your_pretrained_model.h5')
  # some key corresponding may be necessary
  dict_new = language_model_learner.model.state_dict().copy()
  dict_new['key1'] = wgts['key1_pretrained']
  dict_new['key2'] = wgts['key2_pretrained']

  language_model_learner.model.load_state_dict(dict_new)
  language_model_learner.fit(2)
  >>>
  0
  train: [loss value 1 for training set, tensor(metric value 1 for training set, device='cuda:0')]
  valid: [loss value 1 for validation set, tensor(metric value 1 for validation set, device='cuda:0')]
  1
  train: [loss value 2 for training set, tensor(metric value 1 for training set, device='cuda:0')]
  valid: [loss value 2 for validation set, tensor(metric value 1 for validation set, device='cuda:0')]

  # save your model if you are satisfied with its performance
  torch.save(language_model_learner.model.state_dict(), 'your_model_path.pkl')
```

##### Methods to freeze of unfreeze layers
* `ulangel.utils.learner.freeze_all` is a method that sets `requires_grad` of all parameters of the neural network as `Fasle`.
* `ulangel.utils.learner.unfreeze_all` is a method that sets `requires_grad` of all parameters of the neural network as `True`.
* `ulangel.utils.learner.freeze_upto` freezes first n layers of the neural network with `requires_grad` as `False` and `requires_grad` of the rest of layers as 'True'. It's useful when you want to train just the last few layers of a neural network.
Here is the notebook to show how to use these methods: [freeze_to.ipynb](doc/freeze_to.ipynb)

## Software Requirements
Python 3.6
torch 1.3.1
torchvision 0.4.2

## Related efforts
* `Regularizing and Optimizing LSTM Language Models` by Stephen Merity et al
  Article: https://arxiv.org/pdf/1708.02182.pdf
  Github: https://github.com/salesforce/awd-lstm-lm

* `Universal Language Model Fine-tuning for Text Classification` by Jeremy Howard et al
  Article: https://arxiv.org/pdf/1801.06146.pdf
  Github: https://github.com/fastai/course-v3
