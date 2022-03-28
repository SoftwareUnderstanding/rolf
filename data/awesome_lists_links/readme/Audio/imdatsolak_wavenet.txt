# WaveNet implementation in Keras2
Based on https://deepmind.com/blog/wavenet-generative-model-raw-audio/ and https://arxiv.org/pdf/1609.03499.pdf.

This is the based on [Keras WaveNet implementation](https://github.com/basveeling/wavenet/) for Keras 2 and Tensorflow.

I'm currently working on making the single ```mlwavenet.py``` multi-GPU-capable using Horovod, but this has not been fully tested yet, but it seems to work, though there is currently *no support* for predicting with multiple GPUs. I may add it over time...

I use the following command to train on my DUAL-GPU (NVidia GeForce 1080 Ti) using Horovod & OpenMPI:
    
    /usr/local/bin/mpirun -np 2 -H localhost:2 -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -mca btl_tcp_if_exclude eno1 python mlwavenet.py -c multi_gpu_settings.ini -m

The ``-mca btl_tcp_if_exclude eno1`` just means that OpenMPI should not listen on that interface as that one is not configured on my machine...

The parameter  ``-m`` is important as it tells ``mlwavenet.py`` that it will be running in the multi-GPU mode!

Please check out [Horovod for details](https://github.com/uber/horovod)

[Listen to a sample 🎶!](http://data.m-ailabs.bayern/data/Samples/Music/)

## Installation:
Activate a new python2 virtualenv (recommended):

    pip install virtualenv
    mkdir ~/virtualenvs && cd ~/virtualenvs
    virtualenv wavenet
    source wavenet/bin/activate

Clone and install requirements.

    cd ~
    git clone https://github.com/imdatsolak/keras2-wavenet.git
    cd wavenet
    pip install -r requirements.txt

## Dependencies:
This implementation does not support python3 as of now.

## Sampling:
Once the first model checkpoint is created, you can start sampling.

Run:

    $ python2 mlwavenet.py -c <config-file-used-for-training> -C predict -l 5

The latest model checkpoint will be retrieved and used to sample. The sample will be streamed to `[model_dir]/samples`, you can start listening when the first sample is generated.
You can either define the sample_length in settings-file or provide it as parameter (-l) - in seconds

Alternatively, you can specify an epoch to use for generating audio:
    
    $ python2 mlwavenet.py -c <config-file> -C predict -l <duration-seconds> -e <epoch-no>

### Sampling options:
- `predict_length`: float. Number of seconds to sample (length in seconds).
- `sample_argmax`: `True` or `False`. Always take the argmax
- `sample_temperature`: `None` or float. Controls the sampling temperature. 1.0 for the original distribution, < 1.0 for less exploitation, > 1.0 for more exploration.
- `sample_seed`: int: Controls the seed for the sampling procedure.
- `initial_input`: string: Path to a wav file, for which the first `fragment_length` samples are used as initial input.

e.g.:

    $ python2 mlwavenet.py -c <settings-file> -C predict -l 5

## Training:

For training, you now need to create a ```configuration-file```. The file is the Windows(r) .ini file-format. An example is provided.

The default setting is fine. You can immediately start training with it. The settings are for 4.4KHz-training.

    $ python mlwavenet.py -c settings.ini -C train

If you don't provide a '-C' command-line option at all, it is assumed to be 'train'. Normally, it will automatically resume training at the last epoch if the settings-file is the same. If you want to re-start training, you can either provide '-R' (reset) or delete the models-directory.

You can, at any time, stop it using CTRL-C.

## Using your own training data:
- Create a new data directory with a train and test folder in it. All wave files in these folders will be used as data.
    - Caveat: Make sure your wav files are supported by scipy.io.wavefile.read(): e.g. don't use 24bit wav and remove meta info.
- Run with: `$ python2 mlwavenet.py -c <settings-file>`

## Todo:
- [ ] Local conditioning
- [ ] Global conditioning
- [x] Training on CSTR VCTK Corpus
- [x] CLI option to pick a wave file for the sample generation initial input. Done: see `predict_initial_input`.
- [x] Fully randomized training batches
- [x] Soft targets: by convolving a gaussian kernel over the one-hot targets, the network trains faster.
- [ ] Decaying soft targets: the stdev of the gaussian kernel should slowly decay.


## Note on computational cost:
The Wavenet model is quite expensive to train and sample from. We can however trade computation cost with accuracy and fidility by lowering the sampling rate, amount of stacks and the amount of channels per layer.

Configuration: 2x GeForce 1080 Ti (each: 11GiB and ~11TFLOPS), Intel Core i7-6950X CPU @ 3.00GHz (Overclocked: 4.2GHz), 128GiB RAM, 1TB NVME SSD
- Training at 22KHz, about 27 minutes of audio files: 6.5 hrs / epoch
- Prediction of 5 seconds, @ 22KHz: 11 minutes

Deepmind has reported that generating one second of audio with their model takes about 90 minutes.

## Disclaimer
This is a re-implementation of the model described in the WaveNet paper by Google Deepmind. This repository is not associated with Google Deepmind.
# wavenet
