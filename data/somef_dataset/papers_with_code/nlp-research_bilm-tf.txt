# nlp-research/bilm-tf

This repository supports both (1) training ELMo representations and (2) using pre-trained ELMo representaions to your new model

## Installing packages for training
```
pip install tensorflow-gpu==1.2 h5py hgtk
```

## Insatlling packaages for using pre-trained ELMo
```
pip install allennlp hgtk
```

## Using pre-trained ELMo representatinos to your new model
See `usr_dir/embed_with_elmo.py` for detailed example.
Make sure to set `n_characters=262` example during prediction in the `options.json`.
See [here](https://github.com/allenai/bilm-tf#whats-the-deal-with-n_characters-and-padding).
```python
from allennlp.commands.elmo import ElmoEmbedder
import hgtk
import preprocess

options_file = 'path/to/options.json' # Make sure to set n_characters=262
weight_file = 'path/to/weights.hdf5'

elmo = ElmoEmbedder(options_file, weight_file) # create your ELMo class based on weight and option file

sentences = ['밥을 먹자', 'apple은 맛있다']
# normalize, split emj to jaso, add bio tag through preprocess.preprocess_and_tokenize()
preprocessed_sentences = []
for sentence in sentences:
    preprocessed_sentences.append(preprocess.preprocess_and_tokenize(sentence))
#[['Bㅂㅏㅂ', 'Iㅇㅡㄹ', 'Bㅁㅓㄱ', 'Iㅈㅏ'], ['BＡ', 'Iㅇㅡㄴ', 'Bㅁㅏㅅ', 'Iㅇㅣㅆ', 'Iㄷㅏ']]

# get ELMo vectors
vectors = elmo.embed_batch(preprocessed_sentences)

# return value 'vectors' is list of tensors.
# Each vector contains each layer of ELMo representations of sentences with shape (number of sentences, number of tokens(emjs), dimension).
# use elmo.embed_senteces(preprocessed_sentences) to return generator instead of list
```

## Training new ELMo model
Launch docker container if the docker is not launched. (Only Once)
```bash
cd /path/to/usr_dir/scripts
./run_docker.sh
```

Install system packages and set datetime and timezone. Run this script inside the docker. (Only Once)
```bash
docker attach elmo # if you are not inside of docker
cd /path/to/usr_dir/scripts
./install_packages.sh
```

Inside the docker, set hyperparameters by editing code in [train_elmo.py](https://github.com/nlp-research/bilm-tf/blob/master/bin/train_elmo.py)

Edit [train.sh](https://github.com/nlp-research/bilm-tf/blob/master/usr_dir/scripts/train.sh) to set model name (model directory), vocab file path, train file path.
Before training, make sure to convert data files from raw format to train format. See [build_data.sh](https://github.com/nlp-research/bilm-tf/blob/master/usr_dir/scripts/build_data.sh)

Run train.sh inside the docker Print stream to nohoup file for logging (Recommanded).
```bash
cd /path/to/usr_dir/scripts
nohoup ./train.sh &
```

## Converting triained model to hdf5 file
Either inside or outside of docker, edit and run [dump.sh](https://github.com/nlp-research/bilm-tf/blob/master/usr_dir/scripts/dump.sh) to convert trained model to hdf5 file
```bash
cd /path/to/usr_dir/scripts
./dump.sh
```
<i>NOTE</i>: Check your model path in `/path/to/usr_dir/model/model_name/checkpoint` if error occurs when running dump.sh to convert trained model to hdf5 file.