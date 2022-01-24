## Installing packages
I tested this project with python 3.6.9
with the following packages and versions:
tensorflow==1.14.0
gast==0.2.2 (note that you have to install this specific version and overwrite the one installed by tensorflow, o/w there will be errors)

You can install all the packages I used (for running on CPU) using `pip install -r requirements-cpu.txt`, preferrablely under using a virtual environment. 

If you want to use GPU, use `pip install -r requirements.txt`. Depending on your CUDA version, this may require additional setups.

## Data Input
All datas are stored in a `.tsv` (tab separated text file) file. Each row is an entry, and each entry has 3 items (separated by tab `\t`):
* guid: an unique ID for this entry. This is not used in training as a feature.
* label: the real label of this entry.
* sentence: the sentnece of this entry.

AIMed corpus is already split into training and testing sets. They can be viewed in the `AIMed` directory.

This directory contains 2 files: [train.tsv](AIMed/train.tsv) and [dev.tsv](AIMed/dev.tsv). Note that labels for testing data are not used in the prediction process, but rather used for calculating accuracy statistics: precision, recall, etc..

For running the code on BioInfer, you'll need to create a directory named BioInfer and split BioInfer data in `processed_corpus/BioInfer.tsv` into train/dev sets manually.

### Change to other types of data
See `PPIProcessor` in [utilities.py](utilities.py) as a template. What you need to change:
1. in `get_labels()` function: change the return value to the list of all the labels of your sentenfes;
2. in `_create_examples()` function, in the `for` loop, implement how you would get:
    * `guid`: the unique ID of the sentence
    * `text_a`: the text of the sentence
    * `text_b`: leave it to `None`
    * `label`: the label of the sentence.

Finally, in [run_classifier.py](run_classifier.py), change all the instances of `PPIProcessor` to the name of your own data processor.

## Change model size

Only bert-tiny is included in this repository. If you wish to use larger BERT models, download them from [official repo](https://github.com/google-research/bert), and then change `PRETRAIN_DIR` and `PRETRAINED_MODEL` variables.

## Instance Model vs. Sentence Model

### Sentence Model
For example, let the tokenized sentence be:
``[CLS] PROT1 interacts with PROT2 . [SEP]`` 

Sentence model takes the output of token `[CLS]` from the transformer as the input to the classification layers.

Sentence Model is implemented in [sentence_model.py](sentence_model.py).

### Instance Model
Instance Model joins all the representations from tokens indexed with `entity_mask`, (aka token `[CLS]`, `PROT1`, and `PROT2`) and use this long output as input to the classification layers.

The `entity_mask` for the above sentence is `[0, 1, 4]`, where 1 and 4 are the positions of `PROT1` and `PROT2` in the sentence.

Note that if you use instance model, you would implemented the way to calculate `entity_mask` yourself. The function is `get_entity_mask()`. It takes 2 input parameters:
1. `tokens`: this is the list of tokens after tokenization.
2. `tokenizer`: this is the tokenizer used for tokenizing the sentence.

Instance model is implemented in [instance_model.py](instance_model.py).

## Model Usage

Both the scripts mentioned below should be able to run after you changed `PROJECT_DIR`.

Refer to the [BERT repo](https://github.com/google-research/bert) for additional pretrained models, as well as [BioBert repo](https://github.com/dmis-lab/biobert) for models pre-trained with bio text.

### Train the model

First, use `chmod +x *.sh` to make all shell files executable.

Run [fine_tune.sh](fine_tune.sh) to train the model. Please refer to the file itself for documentations.

You can refer to the original bert readme file [BERT_README.md](BERT_README.md) for suggested values.

### Use trained model to predict
Run [predict.sh](predict.sh) to predict. Please refer to the file itself for documentations. 
`test_results.csv` will be generated under directory `TRAINED_CLASSIFIER` after prediction. It's a tsv file with 4 columns:
1. guid
2. predicted probabilities for each category (Note: multiple categories not tested)
3. real label for the sentence
4. the sentence itself

### Change prediction metrics
If you wish to change prediction metrics (e.g., precision, recall), go to `metric_fn` function in both [run_classifier.py](run_classifier.py) and [run_predict.py](run.predict.py) and modify the metrics. Use [this documentation](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/metrics) for the available metrics.