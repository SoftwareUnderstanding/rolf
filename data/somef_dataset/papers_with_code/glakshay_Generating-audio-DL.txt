# What is WaveNet?

WaveNet is a machine learning architecture used for audio generation. Instead of utilizing RNNs, WaveNet uses dilated convolutions to train. 

This project reimplements the paper in Tensorflow with Keras backend. 

See paper and blog for more information:
https://deepmind.com/blog/article/wavenet-generative-model-raw-audio
https://arxiv.org/pdf/1609.03499.pdf

Also, included is our project paper for our work.

## Necessary Tools:
1. Python 3
2. Docker (Docker Engine API v1.40 for gpu)

Works on all platforms but tested on Ubuntu 18

## Building and Running

#### Pull docker image
docker pull tensorflow/tensorflow:2.1.0-gpu-py3
#### Build Code
docker build -t wavenet/latest .
#### Run code
docker run -v $(pwd)/saved_data:/saved_data:rw --gpus all -it --rm --name wavenetbox wavenet/latest 


