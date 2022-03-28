# wavenet
WaveNet vocoder implementation for speech synthesis task

Papers:
* <https://arxiv.org/pdf/1609.03499.pdf>

* <https://arxiv.org/pdf/2011.10469v1.pdf>

Data:

* <https://keithito.com/LJ-Speech-Dataset/>

## Docker

Build container:

`./docker/build.sh <container>`

Run container:

`./docker/run.sh <container> <port>`

Stop container:

`./docker/stop.sh <container>`

## Model utilization

Init project module:

`./scripts/init_module.sh`

Download training data:

`./scripts/download_data.sh`

Download model checkpoint:

`./scripts/download_model.sh`

Start training process:

`./scripts/train_model.sh`

Model inference, this one is configured to process `example/spectrogram.wav` file. Output audio is saved in `example/generated.wav` file:

`./scripts/test_model.sh`
