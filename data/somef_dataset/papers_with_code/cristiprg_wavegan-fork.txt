# GAN-initialization

Prepare to be initialized!

This repo builds on top of [wavegan](https://github.com/chrisdonahue/wavegan) as of June 2018. Since then, they have made several changes and brought several improvements, so I recommend exploring a bit that repo first.

The simplest way to get started with GAN-init is to use the built-in SC09 dataset.

1. Create moments file:
```
python train_specgan.py moments <TRAIN_DIR> \
	--data_dir <DATA_DIR> \
	[--exclude_class <LABEL>] \
	--data_moments_fp <DATA_MOMENTS_FPS>
```

This computes mean and stddev for each frequency bin (feature) and saves them into pickle file. You have the option to exclude a label or not.

2. Train a GAN on specific amount of _data_, for a specific number of _epochs_, _excluding_ or not any label, and save output to a text file:
```
python ./train_specgan.py train <TRAIN_DIR>  \
	--data_dir <DATA_DIR> \	
	--data_moments_fp <DATA_MOMENTS_FPS> \
	[--exclude_class <LABEL> \]
	[--stop_at_global_step <NR_GLOBAL_STEPS> \]
	|& tee -a <LOG_FILE>
```

`--exclude_class` is an optional parameter. The values are numbers 0 to 9, case in which it will just ignore the data points with that specific label. Be sure to check the logs, the system prints out this piece of information in the beginning, in the [loader](https://github.com/cristiprg/wavegan-fork/blob/master/loader.py#L122).
It can also take the value -1, to remove 10% of the training points [uniformly](https://github.com/cristiprg/wavegan-fork/blob/master/train_specgan.py#L144).

Saving the output to a log file is also optional.

3. Train a CNN that loads the weights from the GAN's discriminator:
```
python ./train_cnns.py \
	--gan_type <GAN_TYPE> \
	--data_moments_fp <DATA_MOMENTS_FPS> \
	--data_dir <DATA_DIR> \
	--train_dir <GAN_TRAIN_DIR> \	
	--discard_first_data_percentage <PERCENTAGE_TO_DROP> \
	[--predictions_pickles_directory <PICKLE_PREDICTIONS> \]
	|& tee -a <LOG_FILE>
```

This script loads the GAN model saved in GAN_TRAIN_DIR and performs hyperopt. It uses the training data in DATA_DIR, from which it discards the first PERCENTAGE_TO_DROP percentage. This is useful if you want to train your GAN on the first q% of the data and the CNN on the last (100-q)%, thus partitioning the data into two partitions.

Please also have a look at the descriptions of the other parameters [here](https://github.com/cristiprg/wavegan-fork/blob/master/train_cnns.py#L474-L491).

For example, you can also set `--train_data_percentages` to use only a percentage of the CNN data for training. Thus, the final number of data points will be `train_data_percentages / 100 * (100-q) / 100`. This is useful for getting the data for reproducing Fig. 4.6, where q = 60, (100-q) = 40 and train_data_percentages = 25,50,75,100.

You can also check out the `checkpoint_iters` variable. In case there are multiple GAN models checkpoints saved, you can specifiy which models you would like to load the weights from. Useful for creating the data for figure 4.2.

4. In case you would also like the best resulted model from step 3 (after hyperopt), you can set `perform_test_best_hyperopt = True` [here](https://github.com/cristiprg/wavegan-fork/blob/master/train_cnns.py#L47). This will cause [after hyperopt](https://github.com/cristiprg/wavegan-fork/blob/master/train_cnns.py#L576) to decode the optimized parameters and pass them to the [test script](https://github.com/cristiprg/wavegan-fork/blob/master/test_cnn.py). This test script trains a model with the given architecture and evaluates it against the given test set.

I strongly recommed using the LOG_FILE here. This log file contains general logs as well as the [results](https://github.com/cristiprg/wavegan-fork/blob/master/test_cnn.py#L282). This makes the entire system less complicated to debug and to maintain. To extract the results, simple bash commands to parse the text file do the job.

```
cat <LOG_FILE> | grep Result > results.txt
```
Here, I used the word ["Result"](https://github.com/cristiprg/wavegan-fork/blob/master/test_cnn.py#L282) because it discriminates between the final results and rest of system logs. 

5. If you would like to use another dataset, please follow the Donahue's instructions below to create build the datasets in the .tfrecord format. It is a good practice to create one set of multiple shards for _train_, one for _valid_ and one for _test_. Check the `--name` parameter. Also, make sure to set the [data-constants](https://github.com/cristiprg/wavegan-fork/blob/master/util.py#L12-L19) appropriately.

For experimenting with WaveGAN, the procedures are the same, expect the moments file is not necessary anymore and GAN_TYPE is now `wavegan` instead of `specgan`.

Below this row is the original content of Donahue's wavegan repo.

# WaveGAN

<img src="static/wavegan.png"/>
<img src="static/results.png"/>

Official TensorFlow implementation of WaveGAN (Donahue et al. 2018) ([paper](https://arxiv.org/abs/1802.04208)) ([demo](https://chrisdonahue.github.io/wavegan/)) ([sound examples](http://wavegan-v1.s3-website-us-east-1.amazonaws.com)). WaveGAN is a GAN approach designed for operation on raw, time-domain audio samples. It is related to the DCGAN approach (Radford et al. 2016), a popular GAN model designed for image synthesis. WaveGAN uses one-dimensional transposed convolutions with longer filters and larger stride than DCGAN, as shown in the figure above.

## Usage

### Requirements

```
# Will likely also work with newer versions of Tensorflow
pip install tensorflow-gpu==1.4.0
pip install scipy
pip install matplotlib
```

### Build datasets

You can download the datasets from our paper bundled as TFRecords ...

- [Speech Commands Zero through Nine (SC09)](https://drive.google.com/open?id=1qRdAWmjfWwfWIu-Qk7u9KQKGINC52ZwB) alternate link: [(raw WAV files)](http://deepyeti.ucsd.edu/cdonahue/sc09.tar.gz)
- [Drums](https://drive.google.com/open?id=1nKIWosguCSsEzYomHWfWmmu3RlLTMUIE)
- [Piano](https://drive.google.com/open?id=1REGUUFhFcp-L_5LngJp4oZouGNBy8DPh) alternate link: [(raw WAV files)](http://deepyeti.ucsd.edu/cdonahue/mancini_piano.tar.gz)

or build your own from directories of audio files:

```
python data/make_tfrecord.py \
	/my/audio/folder/trainset \
	./data/customdataset \
	--ext mp3 \
	--fs 16000 \
	--nshards 64 \
	--slice_len 1.5 \
```

### Train WaveGAN

To begin (or resume) training

```
python train_wavegan.py train ./train \
	--data_dir ./data/customdataset
```

If your results are unsatisfactory, try adding a post-processing filter with `--wavegan_genr_pp` or removing phase shuffle with `--wavegan_disc_phaseshuffle 0`. 

To run a script that will dump a preview of fixed latent vectors at each checkpoint on the CPU

```
export CUDA_VISIBLE_DEVICES="-1"
python train_wavegan.py preview ./train
```

To run a (slow) script that will calculate inception score for the SC09 dataset at each checkpoint

```
export CUDA_VISIBLE_DEVICES="-1"
python train_wavegan.py incept ./train
```

To back up checkpoints every hour (GAN training will occasionally collapse)

```
python backup.py ./train 60
```

### Train SpecGAN

Compute dataset moments to use for normalization

```
export CUDA_VISIBLE_DEVICES="-1"
python train_specgan.py moments ./train \
	--data_dir ./data/customdataset \
	--data_moments_fp ./train/moments.pkl
```


To begin (or resume) training

```
python train_specgan.py train ./train \
	--data_dir ./data/customdataset \
	--data_moments_fp ./train/moments.pkl
```

To run a script that will dump a preview of fixed latent vectors at each checkpoint on the CPU

```
export CUDA_VISIBLE_DEVICES="-1"
python train_specgan.py preview ./train \
	--data_moments_fp ./train/moments.pkl
```

To run a (slow) script that will calculate inception score for the SC09 dataset at each checkpoint

```
export CUDA_VISIBLE_DEVICES="-1"
python train_specgan.py incept ./train \
	--data_moments_fp ./train/moments.pkl
```

To back up checkpoints every hour (GAN training will occasionally collapse)

```
python backup.py ./train 60
```

### Generation

The training scripts for both WaveGAN and SpecGAN create simple TensorFlow MetaGraphs for generating audio waveforms, located in the training directory. A simple usage is below; see [this Colab notebook](https://colab.research.google.com/drive/1e9o2NB2GDDjadptGr3rwQwTcw-IrFOnm) for additional features.

```py
import tensorflow as tf
from IPython.display import display, Audio

# Load the graph
tf.reset_default_graph()
saver = tf.train.import_meta_graph('infer.meta')
graph = tf.get_default_graph()
sess = tf.InteractiveSession()
saver.restore(sess, 'model.ckpt')

# Create 50 random latent vectors z
_z = (np.random.rand(50, 100) * 2.) - 1

# Synthesize G(z)
z = graph.get_tensor_by_name('z:0')
G_z = graph.get_tensor_by_name('G_z:0')
_G_z = sess.run(G_z, {z: _z})

# Play audio in notebook
display(Audio(_G_z[0], rate=16000))
```

### Evaluation

Our [paper](https://arxiv.org/abs/1802.04208) uses Inception score to (roughly) measure model performance. If you would like to compare to our reported numbers directly, you may run [this script](https://github.com/chrisdonahue/wavegan/blob/master/eval/inception/score.py) on a directory of 50,000 WAV files with 16384 samples each.

```
python score.py --audio_dir wavs
```


To reproduce our paper results (9.18 +- 0.04) for the SC09 ([download](http://deepyeti.ucsd.edu/cdonahue/sc09.tar.gz)) training dataset, run

```
python score.py --audio_dir sc09/train  --fix_length --n 18620
```



### Attribution

If you use this code in your research, cite via the following BibTeX:

```
@article{donahue2018wavegan,
  title={Synthesizing Audio with Generative Adversarial Networks},
  author={Donahue, Chris and McAuley, Julian and Puckette, Miller},
  journal={arXiv:1802.04208},
  year={2018}
}
```
