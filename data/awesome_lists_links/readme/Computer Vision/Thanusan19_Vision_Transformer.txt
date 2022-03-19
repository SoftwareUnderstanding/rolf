# Vision Transformer: Diatom Dataset
by SATHIAKUMAR Thanusan and BERNARD Guillaume based on Google's ViT implemtentation.

Note: This repository was forked and modified from [google-research/vision_transformer](https://github.com/google-research/vision_transformer).

## Introduction

This repository is an academical work on a new subject introduced by google researchers called: 
Transformer for Image classification at scale.
We worked at Georgia Tech Lorraine with the DREAM research team, a robotic laboratory, in in order to test this new image classification technique on a diatom dataset.

This technique called Vision Transformer was published in the folowing paper: 
[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929).

Overview of the model given by Google: we split an image into fixed-size patches, linearly embed each of them, add position embeddings, and feed the resulting sequence of vectors to a standard Transformer encoder. In order to perform classification, we use the standard approach of adding an extra learnable "classification token" to the sequence.

## Installation

Make sure you have `Python>=3.6` installed on your machine.

→ Install venv package:
```
apt-get install python3-venv
```
→ Create jax-ViT venv:  
```
python3 -m venv venv/jax-ViT
```
→ Activate venv: 
```
source /venv/jax-ViT/bin/activate
```
→ Upgrade pip before installing required package: 
```
python -m pip install --upgrade pip
```
→ Install required package for jax-ViT into the venv:
``` 
pip3 install -r vit_jax/requirements.txt
```
→ Install jax-GPU version: 
```
pip install --upgrade jax jaxlib==0.1.61+cuda110 -f https://storage.googleapis.com/jax-releases/jax_releases.html
```
(“cuda110” → means cuda v.11.0: change this according to the cuda version in your computer)
→ Clone Github code: 
```
git clone https://github.com/Thanusan19/Vision_Transformer.git
```

For more details on Jax, please check the [Jax GitHub repository](https://github.com/google/jax)
Note that installation instructions for GPU differs slightly from the instructions for CPU.


## Available models

Google provides models pre-trained on imagenet21k for the following architectures:
  - ViT-B/16
  - ViT-B/32
  - ViT-L/16
  - ViT-L/32
  - ViT-H/14
  - R50+ViT-B/16 hybrid model (ViT-B/16 on top of a Resnet50 backbone)

You can find all these models in the following storage bucket:
https://console.cloud.google.com/storage/vit_models/

Download one of the pre-trained model with the following command:
```
wget https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz
```
Google also provide models pre-trained on imagenet21k **and** fine-tuned on imagenet2012.

## How to use the different branches

We ended having multiple branches depending on the use. The first one corresponds to our initial ViT implementation changes, and is capable of training on the diatom dataset and has data augmentation capabilities. The second one is "cnn_model" branch, which was used to test simple convolution and PCA based feature extractor. Finally the third one is "resnet_vit" branch which was used to test the resnet50 model as feature extractor.

Here are the general settings to check in the various implementations:
- First make sure that `FINE_TUNE = True` in order to do fine-tuning
- Set the following parameter in order to enable inference: `INFERENCE = True`. Also set the checkpoint's filepath  in: `params = checkpoint.load('../models/model_diatom_final_checkpoints.npz')`
- If you want to train without the fine-tuned weights, use: `LOAD_FINE_TUNNED_CHECKPOINTS = False`. Also set the checkpoint's filepath: `checkpoints_file_path = "../models/model_diatom_checkpoint_step_6000_with_data_aug.npz"`
- Test a saved checkpoint accuracy by setting the following parameter: `CHECKPOINTS_TEST = True`. Also set the checkpoint's filepath: `  checkpoints_file_path = "../models/model_diatom_final_checkpoints.npz"`
- Choose the ViT model to train on with the `model` parameter. The basic model to use could be `model = 'ViT-B_16'`. See branch specific instructions for more details. 
- Choose the dataset to load, e.g: `DATASET = 2`. Choose the dataset between:
  - `0` for CIFAR-10,
  - `1` for dog and cats,
  - `2` for the diatom dataset.
- Set the batch size and epochs according to your training resources and needs. We recommend the following parameters:
  ```python
  epochs = 100
  batch_size = 256
  warmup_steps = 5
  decay_type = 'cosine'
  grad_norm_clip = 1
  accum_steps = 64  # 64--> GPU3  #8--> TPU
  base_lr = 0.03 # base learning rate
  ```
- If you want to use data augmentation during training, change the `doDataAugmentation` parameter inside the corresponding call to the python data generator `MyDogsCats()`: `doDataAugmentation=True`. We recommend not using data augmentation on the train and validation sets.

Here is an example of parameters we recommend to use if you want to do fine-tuning with untrained fine-tuning weights, on the diatom dataset:
```python
INFERENCE = False
FINE_TUNE = True
LOAD_FINE_TUNNED_CHECKPOINTS = False
CHECKPOINTS_TEST = False
DATASET = 2 #to load diatom dataset
batch_size = 256 #can be set to 512 for no data augmentation and simple ViT model fine-tuning
epochs = 100
warmup_steps = 5
decay_type = 'cosine'
grad_norm_clip = 1
accum_steps = 64  # 64--> GPU3  #8--> TPU
base_lr = 0.03 #base learning rate
```

Once you have checked the specific recommendation for the specific branch, you can launch the training using:
```
cd vit_jax/
python vit_jax.py
```
__NB:__ Make sure you have activated the virtual environment before launching. (e.g. `source /venv/jax-ViT/bin/activate`)

### 1. `main`

- In the file `vit_jax/vit_jax.py`, to train with the basic ViT model we used: `model = 'ViT-B_16'`
- In the file `vit_jax/ViT_python_generator.py`, inside the definition of the `_generator()` method, you can uncomment and change the code for the actual data augmentation of the images. There are still comments of code that could do image translation, rotation, etc.

### 2. `cnn_model`
This branch was used to test different convolution layers and PCA as feature extractor in the ViT hybrid model.
First of all check the following parameters: 
- `model = 'ViT-B_16'`
- Set `SAVE_IMG = True` to save feature extractor's input and output.
Then activate the code associated to the wanted feature extractor to `True` in `vit_jax/vit_jax.py` file.

### 3. `resnet_vit`

This branch was used to test Resnet50 model as feature extractor in the ViT hybrid model.
Check if the following parameters are set in "vit_jax/vit_jax.py": 
- `model = 'R50+ViT-B_16'`
- `doDataAugmentation=False`
