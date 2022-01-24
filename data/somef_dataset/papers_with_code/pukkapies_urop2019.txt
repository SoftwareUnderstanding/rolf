# Deep Learning for Music Tagging (aka "Orpheus")

This is the repository of an Imperial College UROP 2019 project in deep learning for music tagging. We aimed to develop an end-to-end music auto-tagger competitive with the state-of-the-art. We replicated the convolutional neural network architecture proposed by (Pons, et al., 2018) in [this](https://arxiv.org/pdf/1711.02520.pdf) paper, and reproduced the results they obtained on the [Million Song Dataset](http://millionsongdataset.com/). 

Since our model learned to predict some audio features quite accurately, we decided to call it "Orpheus", like the legendary ancient Greek poet and musician.

## Table of Contents

* [Introduction](https://github.com/pukkapies/urop2019#introduction)
* [System Requirements](https://github.com/pukkapies/urop2019#system-requirements)
* [Data Cleaning](https://github.com/pukkapies/urop2019#data-cleaning)
    * [Errors in the .MP3 Audio Files](https://github.com/pukkapies/urop2019#errors-in-the-mp3-audio-files)
    * [Errors in the Dataset](https://github.com/pukkapies/urop2019#errors-in-the-dataset)
    * [Last.fm Tags](https://github.com/pukkapies/urop2019#lastfm-tags)
* [Data Input Pipeline](https://github.com/pukkapies/urop2019#data-input-pipeline)
    * [TFRecords](https://github.com/pukkapies/urop2019#tfrecords)
    * [TFRecords into a tf.data.Dataset](https://github.com/pukkapies/urop2019#tfrecords-into-a-tfdatadataset)
* [Model and JSON Configuration](https://github.com/pukkapies/urop2019#model-and-json-configuration)
* [Training](https://github.com/pukkapies/urop2019#training)
* [Validating & Predicting](https://github.com/pukkapies/urop2019#validating-and-predicting)
* [Results](https://github.com/pukkapies/urop2019#results)
* [Conclusion](https://github.com/pukkapies/urop2019#conclusion)
* [References](https://github.com/pukkapies/urop2019#references)
* [Contacts / Getting Help](https://github.com/pukkapies/urop2019#contacts--getting-help)

## Introduction

This project makes use of the freely-available [Million Song Dataset](http://millionsongdataset.com/), and its integration with the [Last.fm Dataset](http://millionsongdataset.com/lastfm/). The former provides a link between all the useful information about the tracks (such as title, artist or year) and the audio track themselves, whereas the latter contains tags information on some of the tracks. A preview of the audio tracks can be fetched from services such as 7Digital, but this is allegedly not an easy task. 

If you are only interested in our final results, click [here](https://github.com/pukkapies/urop2019#results).

If you want to use some of our code, or try to re-train our model on your own, read on. We will assume you have access to the actual songs in the dataset. Here is the outline of the approach we followed:

1. Extracte all the useful information from the Million Song Dataset and clean both the audio tracks and the Last.fm tags database to produce our final 'clean' data;

2. Prepare a flexible data input pipeline and transform the data in a format which is easy to consume by the training algorithm;

3. Prepare a flexible training script which would allow for multiple experiments (such as slightly different architectures, slightly different versions of the tags database, or different training parameters);

4. Train our model and use it to make sensible tag predictions from a given input audio.

In the following sections, we will provide a brief tutorial of how you may use this repository to make genre predictions of your own, or carry out some further experiments.

## System Requirements

* [Python](https://www.python.org/)* 3.6 or above
* One or more CUDA-enabled GPUs
* Mac or Linux environment
* [TensorFlow](https://www.tensorflow.org/beta)* 2.0.0 RC0 or above (GPU version)
* [H5Py](https://www.h5py.org/) 2.3.1 -- to read the the HDF5 MSD summary 
* [LibROSA](https://librosa.github.io/librosa/)* 0.7.0 + [FFmpeg](https://www.ffmpeg.org/)* -- to read, load and analyse audio files
* [mutagen](https://mutagen.readthedocs.io/en/latest/) 1.42.0 -- to read audio files
* [sounddevice](https://python-sounddevice.readthedocs.io/en/latest/)* 0.3.12 -- to record audio from your microphone through terminal
* [sparse](https://sparse.pydata.org/en/latest/) 0.8.9 -- to perform advanced operations on the tags database (and process data using sparse matrices)
* Other common Python libraries such as [Pandas](https://pandas.pydata.org/) or [NumPy](https://numpy.org/)

If you are just running the lite version of our prediction tool, all you need are the packages marked with *.

## Data Cleaning

### Errors in the .MP3 Audio Files

You can use `fetcher.py` to scan the directory which contains your audio files and store their file path, file size, duration and number of channels in a Pandas dataframe. As it turns out, some files cannot be opened, and some others *can* be opened but are completely (or mostly) silent. To tackle the first issue, `fetcher.py` can automatically purge the faulty tracks from the final output (most of them have either zero or extremely small file size). To tackle the second issue, we made use of LibROSA and its `librosa.effects.split()` function, which splits an audio signal into non-silent intervals.

In order to do so, LibROSA first needs to convert an audio file into a waveform array, stored in NumPy format. The information on audio silence will then be processed by `wrangler_silence.py`, which will remove from the original Pandas dataframe all the tracks which do not satisfy certain user-set criteria. You have two options here: either you use `mp3_to_numpy.py` to create in advance `.npz` files for your entire dataset, or you generate and process the audio arrays on the fly when using `wrangler_silence.py`. 

We originally created `.npz` files for the entire dataset, as we did not have the whole audio cleaning path outlined yet, and we needed to have the data easily accessible for frequent experimentations. We would however recommend against this approach, as creating `.npz` files often requires a huge amounts of storage space.

You will then have a Pandas dataframe (saved into a `.csv` file) containing only the good audio files.

*Example:*

```bash
# save a .csv file containing audio tracks and tracks info
python fetch.py --root-dir /srv/data/msd/7digital /path/to/output/fetcher.csv
```
```bash
# convert audio tracks into .npz
python mp3_to_numpy.py /path/to/output/fetcher.csv --root-dir-npz /srv/data/urop2019/npz --root-dir-mp3 /srv/data/msd/7digital
```
```bash
# save a .csv file containing audio silence info, optionally also discard 'silent' tracks
python wrangler_silence.py /path/to/output/fetcher.csv /path/to/output/wrangler_silence.csv --root-dir-npz /srv/data/urop2019/npz --root-dir-mp3 /srv/data/msd/7digital --min-size 100000 --filter-tot-silence 15 --filter-max-silence 3
```

### Errors in the Dataset

The raw HDF5 Million Song Dataset file, which contains three smaller datasets, are converted into multiple Pandas dataframes. The relevant information is then extracted and merged. According to the MSD website, there are [mismatches](http://millionsongdataset.com/blog/12-2-12-fixing-matching-errors/) between these datasets. To deal with the issue, `wrangler.py` takes a `.txt` file with a list of tids which could not be trusted, and remove the corresponding rows in the dataframe. Furthermore, MSD also provides a `.txt` file with a list of tracks which have [duplicates](http://millionsongdataset.com/blog/11-3-15-921810-song-dataset-duplicates/). To deal with this other issue, `wrangler.py` by default only keeps one version of each duplicate (picked randomly), and removes the rest.

The dataframe from the above paragraph is merged with the dataframe produced by the above audio section followed by removing unnecessary columns to produce the 'ultimate' dataframe, which contains essential information about the tracks that will be used throughout the project.

*Example:*

```
python wrangler.py /path/to/output/fetcher.csv /path/to/ultimate.csv --path-h5 /srv/data/msd/entp/msd_summary_file.h5 --path-db /srv/data/msd/lastfm/lastfm_tags.db --path-txt-dupl /path/to/duplicates.txt --path-txt-mism /path/to/mismatches.txt
```

In order to save storage space and time, a different order of code execution was instead used though.

*Example:*

```
python fetcher.py --root-dir /srv/data/msd/7digital /path/to/output/fetcher.csv
```
```
python wrangler.py /path/to/output/fetcher.csv /path/to/output/wrangler.csv --path-h5 /srv/data/msd/entp/msd_summary_file.h5 --path-db /srv/data/msd/lastfm/lastfm_tags.db --path-txt-dupl /path/to/duplicates.txt --path-txt-mism /path/to/mismatches.txt --discard-dupl --discard-no-tag
```
```
python mp3_to_numpy.py /path/to/output/wrangler.csv --root-dir-npz /output/dir/npz --root-dir-mp3 /srv/data/msd/7digital
```
```
python wrangler_silence.py /path/to/output/wrangler.csv /path/to/output/wrangler_silence.csv --root-dir-npz /output/dir/npz/ --root-dir-mp3 /srv/data/msd/7digital/
```
```
python wrangler_silence.py /path/to/output/wrangler_silence.csv /path/to/ultimate.csv --min-size 200000 --filter-trim-length 15 --filter-tot-silence 3 --filter-max-silence 1
```

This reduces the number of useful tracks from 1,000,000 to ~300,000.

### L<span>ast.f</span>m Tags

#### Performing Queries

The `lastfm.py` module contains three classes, `LastFm`, `LastFm2Pandas` and `Matrix`. The first two classes contain all the basic tools for querying the Lastfm database. The former directly queries the database by SQL, whereas the latter converts the database into dataframes and performs queries using Pandas. In some of the functions in later sections, an input parameter may require you to pass an instance of one of these two classes. The last one, on the other hand, contains some advanced tools to perform a thorough analysis of the tags distribution in the tags database.

*Example:*

To use `LastFm`:

```python
fm = lastfm.LastFm('/srv/data/msd/lastfm/SQLITE/lastfm_tags.db')
```
To use `LastFm2Pandas`:

```python
fm = lastfm.LastFm2Pandas('/srv/data/msd/lastfm/SQLITE/lastfm_tags.db')
```
To use `LastFm2Pandas` from converted `.csv`'s (instead of the original `.db` file):

```python
# generate .csv's
lastfm.LastFm('/srv/data/msd/lastfm/lastfm_tags.db').db_to_csv(output_dir='/srv/data/urop')

# create tags, tids and tid_tag dataframes
tags = pd.read_csv('/srv/data/urop/lastfm_tags.csv')
tids = pd.read_csv('/srv/data/urop/lastfm_tids.csv')
tid_tag = pd.read_csv('/srv/data/urop/lastfm_tid_tag.csv')

# create class instance
fm = lastfm.LastFm2Pandas.load_from(tags=tags, tids=tids, tid_tag=tid_tag)
```

The major difference between the two classes is that `LastFm` is quicker to initiate, but some queries might take some time to perform, whereas `LastFm2Pandas` may take longer to initiate (due to the whole dataset being loaded into the memory). However, `LastFm2Pandas` contains some more advanced methods, and it is reasonably quick to initialise if database is converted into `.csv` files in advance. Moreover, you will need to use `LastFm2Pandas` if you want to perform advanced an advanced analysis of the tags distribution using the `Matrix` class.

Finally, `metadata.py` contains some basic tools to explore the `msd_summary_file.h5` file.

#### Filtering

In the L<span>ast.f</span>m database there are more than 500,000 different tags. Such a high number of tags is clearly useless, and the tags need to be cleaned in order for the training algorithm to learn to make some sensible predictions. The tags are cleaned using `lastfm_cleaning_utils.py` and `lastfm_cleaning.py`, and the exact mechanisms of how they work can be found in the documentation of the scripts.

In brief, the tags are divided into two categories: genre tags, and vocal tags (which in our case are 'male vocalist', 'female vocalist', 'rap' and 'instrumental').

For genre tags, we first obtained a list of tags from the L<span>ast.f</span>m database which have appeared for more than 2000 times, then we manually filtered out the tags that we considered rubbish (such as 'cool' or 'favourite song'). We fed the remaining 'good' tags into `generate_genre_df()`, which searched for similar spelling of the same tag within a 500,000+ tags pool (tags with occurrence ≥ 10), and we produced a handy dataframe with manually chosen tags in one column, and similar matching tags from the pool in the other.

For vocal tags, we first obtained a long list of potentially matching tags for each of the four vocal tags, then we manually separated the 'real' matching tags from the rest, for each of the tag lists. We fed the tag lists into `generate_vocal_df()`, and we produced a dataframe with the structure previously outlined.

Finally, the `generate_final_df()` function merged the two dataframes into one, and passed it to the next script.

To search for similar tags, we did the following:

1. Remove all the non-alphabet and non-number characters and any single trailing 's' from the raw tags with occurance ≥ 10 and the target tags (the classified genre tags and vocal tags). If any transformed raw tag is identical to any target tag, the raw tag is merged into target tag;

2. Repeat the same merging mechanism as 1, but replace '&' with 'n', '&' with 'and' and 'n' with 'and';

3. Repeat the same merging mechanism as 1, but replace 'x0s' with '19x0s' and 'x0s' with '20x0' (without removing the trailing 's'; *x* denotes a number character here).

See [here](https://github.com/pukkapies/urop2019/tree/master/code/msd#tags-cleaning) for more details on how you may tailor the merging mechanism by defining a new filtering function.

If you want to actually see the dataframe which contains all the clean tag info (which will then be used by `lastfm_cleaning.py` to produce the new `.db` file), you can generate it using the `generate_final_df()` functions, which combines all the tools mentioned above, and which allows a lot of room for customization and fine-tuning.

*Example:*

```python
from lastfm_cleaning_utils import generate_final_df

generate_final_df(from_csv_path='/srv/data/urop', threshold=2000, sub_threshold=10, combine_list=[['rhythm and blues', 'rnb'], ['funky', 'funk']], drop_list=['2000', '00', '90', '80', '70', '60'])
```

If you want just to use of our clean dataset, `lastfm_cleaning.py` will automatically make use of tools above to produce a new clean `.db` file. The final database has the same structure as the original`lastfm_tags.db` database. Therefore, it can be queried using the same `lastfm.py` module. 

*Example:*

```
python lastfm_cleaning.py /srv/data/msd/lastfm/SQLITE/lastfm_tags.db /srv/data/urop/clean_lastfm.db
```

Finally, the `.txt` files containing the lists of tags we used in our experiment can be found in [this](https://github.com/pukkapies/urop2019/tree/readme/code/msd/config) folder. 

## Data Input Pipeline

### TFRecords

To store the necessary information we needed for training, we used the TFRecords format. The `preprocessing.py` script does exactly this. In each entry of the `.tfrecord` file, it stores the audio as an array in either waveform or log-mel-spectrogram format. It will also store the track ID to identify each track, and the tags from the clean tags database in a one-hot vector format. It will accept audio as either `.mp3` files, or as `.npz` files where each entry contains the audio as an array and the sample rate. The user can choose the sample rate to store the data in as well as the number of mel bins (when storing the audios as log-mel-spectrogram). The user can also specify the number of  `.tfrecord` files to split the data between.

In our case, we used 96 mel bins, a sample rate of 16kHz and split the data into 100 .`tfrecord` files. We also had the data stored as `.npz` files, since we had previously converted the `.mp3` files into NumPy format for silence analysis. We would once again recommend users to convert directly from `.mp3` files, as the `.npz` files need a lot of storage. 

*Example:*

```bash
python preprocessing.py waveform /output/dir/ --root-dir /srv/data/npz --tag-path /srv/data/urop/clean_lastfm.db --csv-path /srv/data/urop/ultimate.csv --sr 16000 --num-files 100 --start-stop 1 10
```

It is recommended to use [tmux](https://github.com/tmux/tmux/wiki) split screens to speed up the process.

### TFRecords into a tf.data.Dataset

The `projectname_input.py` module was used to create ready-to-use TensorFlow datasets from the `.tfrecord` files. Its main feature is to create three datasets for train/validating/testing by parsing the `.tfrecord` files and extracting a 15 sec window from the audio, then normalizing the data. If waveform is used, we normalized the batch, but if log mel-spectrogram is used, we normalized with respect to the spectrograms themselves (Pons, et al., 2018). The module will also create mini-batches of a chosen size.

The `projectname_input.py` module again leaves a lot of room for customisation. There are functions to exclude certain track IDs from the dataset, to merge certain tags (e.g. 'rap' and 'hip hop'), or to only include some tags. The user can also choose the size of the audio window and whether the window is to be extracted at random, or centred on the audio array. 

In the training script, we use the `generate_datasets_from_dir()` function to automatically use all the `.tfrecord` files in the specified directory. In order to manually generate one or more datasets from a list of `.tfrecord` files, you can use the `generate_datasets()` function.

*Example:*

```python
import lastfm

# instantiate lastfm
lf = lastfm.LastFm('/srv/data/urop/clean_lastfm.db')

# list top ten tags from popularity dataframe
top_tags = lf.popularity()['tags'][:10].tolist()

# list .tfrecord files to parse
files = ['/srv/data/urop/tfrecords-waveform/waveform_1.tfrecord', '/srv/data/urop/tfrecords-waveform/waveform_2.tfrecord']

# create two datasets from two .tfrecord files, use only top ten tags, merge together tags 'pop' and 'alternative'
train_dataset, valid_dataset = generate_datasets(files, audio_format='waveform', split=[1,1,0], which=[True, True, False], with_tags=top_tags, merge_tags=['pop', 'alternative']) # split=[50, 50, 0] would have had the same effect here; or also split=[50, 50] together with which=[True, True]
```

Finally, this data input pipeline is optimised following the [official guidelines](https://www.tensorflow.org/beta/guide/data_performance) for TensorFlow 2.0.

## Model and JSON Configuration

The model we used was designed by (Pons, et al., 2018). See their GitHub [repository](https://github.com/jordipons/music-audio-tagging-at-scale-models) for more details. In our experiment, as mentioned above, we have followed their approach and compared the performance when training with **waveform** or **log mel-spectrogram** audio format. Since the model they provide is written using TensorFlow 1.x syntax, we have rewritten the same model using TensorFlow 2.0. You can find the 'upgraded' model in `projectname.py`.

In brief, `projectname.py` contains two frontend architectures (one for waveform and one for log mel-spectrogram) and a single backend architecture, with a `build_model()` function which combines the two to produce the complete model that will be used for training.

In order to avoid having to manually tinker with the training code every time a training parameter has to be changed, all the training parameters are set through a handy JSON file. You can create an empty `config.json` file by using the `create_config_json()` function. Here is an outline of how the JSON file is structured:

1. `model`: contains parameters to set the number of dense units and convolutional filters in the model;

2. `model-training`: contains important training parameters such as batch size or window length and allow you to fully specify the optimizer to use;

3. `tags`: contains parameters to specify which tags to use when parsing the TFRecords;

4. `tfrecords`: contains parameters to specify how the audio tracks were encoded in the TFRecords such as sample rate or the number of frequency bands in the mel scale.

See the inline comments for the `create_config_json()` function within `projectname.py` for more details. 

*Example:*

```python
import projectname

# to create an empty .json
projectname.create_config_json('/srv/data/urop/config.json')

# to create an empty .json and manually enter some parameters (equivalent to editing the file after creation)
projectname.create_config_json('/srv/data/urop/config.json', 'batch_size'=32)
```

## Training

We have written two separate scripts for the training algorithm, `training.py` 
and `'training_custom.py`. The main difference between the two is that the former makes use of the built-in Keras `model.fit`, whereas the latter makes use of a custom training loop (as described in the [official guidelines](https://www.tensorflow.org/beta/guide/keras/training_and_evaluation#part_ii_writing_your_own_training_evaluation_loops_from_scratch) for TensorFlow 2.0) where each training step is performed manually. While `training.py` only allows the introduction of advanced training features through Keras callbacks, `training_custom.py` allows total flexibility in the features you could introduce. 

Both scripts assume you have one or more GPUs available, and make use of a MirroredStrategy to distribute training. Both scripts write (train and validation) summaries on TensorBoard and save checkpoints at the end of each epoch, and they also have the option to enable early stopping or learning rate reduction on plateau. Only the custom loop implements cyclical learning rate and the one-cycle policy, as described by (N. Smith, 2018) in [this](https://arxiv.org/pdf/1803.09820.pdf) paper.

For ease of use, `projectname_train.py` is wrapper of the two scripts. By default, the custom loop is selected, unless a different choice is specified. You may control all the training parameters by tweaking the `config.json` file.

*Example:*

```bash
# train for 10 epochs on GPUs 0,1 using waveform
python waveform --epochs 10 --root-dir /srv/data/urop/tfrecords-waveform --config-path /srv/data/urop/config.json --lastfm-path /srv/data/urop/clean_lastfm.db --cuda 0 1
``` 
```bash
# train for 10 epochs on GPUs 0,1 using waveform and the custom loop
python waveform --epochs 10 --root-dir /srv/data/urop/tfrecords-waveform --config-path /srv/data/urop/config.json --lastfm-path /srv/data/urop/clean_lastfm.db --cuda 0 1 --custom
``` 

Furthermore, it is possible to stop the scripts in the middle of training by keyboard interrupt and recover from a saved checkpoint using the `--resume-time` parameter.

The `projectname_train.py` script makes use of `projectname_input.py` to generate training and validation datasets. If you want to perform the model training with more flexibility in choosing your own datasets, you may generate your own datasets using the tf.data API and then do the following:

```python
import os
import tensorflow as tf

import training
import projectname_train

strategy = tf.distribute.MirroredStrategy()
#train_dataset = strategy.experimental_distribute_dataset(train_dataset)
#valid_dataset = strategy.experimental_distribute_dataset(valid_dataset)

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

config = projectname_train.parse_config('/srv/data/urop/config.json', '/srv/data/urop/clean_lastfm.db')

training.train(train_dataset, valid_dataset, frontend='waveform', strategy=strategy, config=config, epochs=10)
```
If you prefer to use `training_custom.py`, do exactly the same procedure as above, except replacing `training` with `training_custom` and uncommenting the two `strategy.experimental_distribute_dataset()` lines.

## Validating and Predicting

The evaluation tools are contained in the script `projectname.py`. There is a `test()` function which simply tests the model's performance on the test dataset from a certain checkpoint. There is also a `predict()` function which takes an audio array (in waveform or log mel-spectrogram format) and uses the model to return the most confident tag predicitons for that track. Optionally, the audio array might be sliced in `n_slices` sliding windows of length `window_length`, and the final tag predictions will average out the tag predictions for each single slice. In either case, you will need to pass a `threshold` to determine which tags are shown, based on their prediction confidence.

*Example:*

To test a log-mel-spectrogram model on the test dataset (as specified by `split` in the config JSON):

```
python projectname.py test log-mel-spectrogram --checkpoint /path/to/model/checkpoint --config /path/to/config.json --lastfm /path/to/clean/lastfm.db --tfrecords-dir /srv/data/urop/tfrecords-log-mel-spectrogram
```

To use the same model to predict tags with threshold 0.1 for a single audio track (or multiple tracks in the same folder):

```
python projectname.py predict log-mel-spectrogram --checkpoint /path/to/model/checkpoint --config /path/to/config.json --lastfm /path/to/clean/lastfm.db -t 0.1 --mp3 /path/to/your/song.mp3
```

To use the same model to predict tags with threshold 0.1 for a 30 sec recording:

```
python projectname.py predict log-mel-spectrogram --checkpoint /path/to/model/checkpoint --config /path/to/config.json --lastfm /path/to/clean/lastfm.db -t 0.1 --record --record-length 30
```

## Results

Here are our results when performing different experiments using both waveform and log-mel-spectrogram. We always trained on the top 50 tags from our clean L<span>ast.f</span>m database. 

Here are the exact tags we used, ordered by popularity: 'rock', 'female vocalist', 'pop', 'alternative', 'male vocalist', 'indie', 'electronic', '00s', 'rnb', 'dance', 'hip-hop', 'instrumental', 'chillout', 'alternative rock', 'jazz', 'metal', 'classic rock', 'indie rock', 'rap', 'soul', 'mellow', '90s', 'electronica', '80s', 'folk', 'chill', 'funk', 'blues', 'punk', 'hard rock', 'pop rock', '70s', 'ambient', 'experimental', '60s', 'easy listening', 'rock n roll', 'country', 'electro', 'punk rock', 'indie pop', 'heavy metal', 'classic', 'progressive rock', 'house', 'ballad', 'psychedelic', 'synthpop', 'trance' and 'trip-hop'.

### Experiment 1

This experiment was used to try to replicate the results by (Pons, et al., 2018), and compare the performance obtained on our dataset using waveform and log-mel-spectrogram. We ran this experiments using a constant learning rate of 0.001.

![alt text](https://github.com/pukkapies/urop2019/blob/master/waveform.png)

![alt text](https://github.com/pukkapies/urop2019/blob/master/log-mel-spectrogram.png)

|                                            | AUC-ROC |  AUC-PR |
| ------------------------------------------ |:-------:|:-------:|
| Waveform (ours)                        	 | 86.96   | 39.95   |
| Log-mel-spectrogram (ours)                 | 87.33   | 40.96   |
| Waveform (Pons, et al., 2018)              | 87.41   | 28.53   |
| Log-mel-spectrogram (Pons, et al., 2018)   | 88.75   | 31.24   |


The exact parameters we have used can be found [here](https://github.com/pukkapies/urop2019/blob/master/waveform_config.json) (waveform) and [here](https://github.com/pukkapies/urop2019/blob/master/log-mel-spectrogram_config.json) (log-mel-spectrogram).

### Experiment 2

This experiment was used to test the effectiveness of cyclic learning rate (Smith, 2018) as well as an attempt to try and improve the model. We ran this experiment on an identical run of the log-mel-spectrogram above, using cyclic learning rate varying linearly between 0.0014/4 instead of a constant learning rate of 0.001.

![alt text](https://github.com/pukkapies/urop2019/blob/master/cyclic-learning-rate.png)

![alt text](https://github.com/pukkapies/urop2019/blob/master/log-mel-spectrogram-cyclic.png)

|                                            | AUC-ROC |  AUC-PR |
| ------------------------------------------ |:-------:|:-------:|
| Log-mel-spectrogram (ours)                 | 87.33   | 40.96   |
| Log-mel-spectrogram (ours, cyclic lr)      | 87.68   | 42.05   |
| Log-mel-spectrogram (Pons, et al., 2018)   | 88.75   | 31.24   |

The exact parameters we have used can be found [here](https://github.com/pukkapies/urop2019/blob/master/log-mel-spectrogram-cyclic_config.json).

## Conclusion

In general, we can see that training the MSD dataset on log mel-spectrogram has a better performance than training on waveform, which agrees with the result produced by (Pons, et al., 2018). Note that (Pons, et al., 2018) suggests that when the size of the dataset is large enough, the quality difference between waveform and log-mel-spectrogram model is insignificant (with 1,000,000+ songs).

In our experiments, we have also cleaned the L<span>ast.f</span>m database by removing tags which are more subjective or have vague meaning, which was not done in (Pons, et al., 2018). According to the results above, the AUC-PR of both waveform and log-mel-spectrogram has significantly improved from (Pons, et al., 2018), while in the meantime maintaining comparable AUC-ROC. 

We have therefore shown that training the model using cleaner tags improves the quality of the model.

In our experiments, we also tried to apply the 'disciplined approach to neural network hyper-parameters' techniques outlined in (Smith, 2018), and clearly obtained much better results on our validation dataset.

We have therefore also confirmed that mindfully varying the learning rate throughout the training indeed results in better quality of the model.

## References

Pons, J., Nieto, O., Prockup, M., Schmidt, E., Ehmann, A., Serra, X.: END-TO-END LEARNING FOR MUSIC AUDIO TAGGING AT SCALE. Proc. of the 19th International Society for Music Information Retrieval Conference (ISMIR). Paris, France (2018)

Smith, L. N.: A DISCIPLINED APPROACH TO NEURAL-NETWORK HYPER-PARAMETERS: PART 1 – LEARNING RATE, BATCH SIZE, MOMENTUM, AND WEIGHT DECAY. arXiv preprint [arXiv:1803.09820](https://arxiv.org/pdf/1803.09820.pdf) (2018)

## Contacts / Getting Help

calle.sonne18@imperial.ac.uk

chon.ho17@imperial.ac.uk

davide.gallo18@imperial.ac.uk / davide.gallo@pm.me

kevin.webster@imperial.ac.uk
