# AdaIN-TensorFlow2
AdaIN(from the paper Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization, https://arxiv.org/abs/1703.06868) implementation with TensorFlow 2

Original PyTorch code: https://github.com/naoto0804/pytorch-AdaIN

Includes TFLite conversion for mobile/embedded usage.

## Requirements
* tensorflow >= 2.0.0
* tensorflow_datasets

## Note
* **Most of the code is from naoto0804/pytorch-AdaIN https://github.com/naoto0804/pytorch-AdaIN**.
* This code was written for studying, so the code may be hard to understand. I'll try my best to improve code readability.

## Usage
### Download style images
```
cd STYLE/IMAGE/DIRECTORY/
wget http://web.fsktm.um.edu.my/~cschan/source/ICIP2017/wikiart.zip
unzip wikiart.zip
```
### Train
```
python3 train.py -exp_name EXP_NAME [-lr LEARNING_RATE] [-batch_size BATCH_SIZE] [-output_dir OUTPUT_DIR] [-save_every SAVE_EVERY] [-save_tflite SAVE_TFLITE]
```
Check out the code for more training options.

## TODO
* Generate TFLite model from checkpoints
* Color preserving
* Style interpolation
* Different style for different region