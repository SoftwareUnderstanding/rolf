# noplants - an ANN based weed-recognition software

## Setting up a conda environment
In ourder to use our code certain dependencies must be met. We propose setting up a conda environment.
Pleas install Miniconda on Linux according to the following link: https://conda.io/projects/conda/en/latest/user-guide/install/linux.html

All the needed dependencies are listed in the environment.yml file and a respective environment can be created with it.
In order to do so, please first clone the git environment and then open a terminal and go to the respective directory. Then type:

```console
(base) username@dev:~/noplants$ conda env create -f environment.yml # creates the sepecified environment
(base) username@dev:~/noplants$ conda actvate killingplants # now code can be executed
(killingplants) username@dev:~/noplants$ conda deactivate # deactivates environment
(base) username@dev:~/noplans$
```

## Prepare targets
In order to make most of the data availabe we set up a script preparing the data for our network implementation.
The prepare_targets.py takes high resolution pitures and their high resolution labes, crobs them into small subimages and transform the targets so that the three colour channels of resulting RGB image give the probability for belonging to a certan class (good plant, weed, ground).

We train our network on segmentation as well as stem prediction. Thus, two kinds of targets have to be created, the segmentation and the root targets.

Thus, the script hast to be excecuted two times, once for the stem and once for the segmentation data. To run the script please adjust the following parameter in the hyperparameters.py:
```python
# Data Preparation
ORIGIN_LBL_DIRECTORY = 'stem_lbl_human' # folder with rare data
ORIGIN_DATA_DIRECTORY = 'stem_data'  # folder with labeled data
SUBPICS = 200
CROP_SIZE = (256, 256, 3)
# please create following directory
SAVE_LBL = 'stem_lbl_cropped_container/stem_lbl_cropped/'
SAVE_DATA = 'stem_data_cropped_container/stem_data_cropped/'
```
After creating the saving paths as specified run the sript as follows for stem targets:
```console
(killingplants) usr@dev:~/noplants$ python prepare_targets.py
```
When creating segmentation targets, adjust the hyperparameters so that they fit your segmentation data and pleacse include the argument 'segmentation' in the shell command:
```console
(killingplants) usr@dev:~/noplants$ python prepare_targets.py segmentation
```

## Data pipeline
The Data Pipeline does not has to be executed on its own but is used by the Netw when Training. It takes pictures from a specified Directory and applies varies sorts of data augmentation specified by
```python
self.data_gen_args = dict(....)
```
We use flipping and zooming on both and changing brightness only on the input data.
Since we use the ImageDataGenerator() by keras, make sure that you specifiy the input path correctly: you have to build a contaner containing another dircetory with the actual images (See **Prepare targets**).

## Training
To run the training adjust the following parameters in the hyperparameters.py script:

```python
# Training
MODEL_SAVE_DIR = # directory where you want to save your models to
DATA_TRAIN_STEM_LBL = 'stem_lbl_cropped_container' # please be aware in the container needs to be another folder with the actual data
DATA_TRAIN_STEM = 'stem_data_cropped_container'
DATA_TRAIN_SEG_LBL = 'seg_lbl_cropped_container' # please be aware in the container needs to be another folder with the actual data
DATA_TRAIN_SEG = 'seg_data_cropped_container'
SAVE_STEPS = 100 # after how many training steps a model should be saved, don't go lower than 100
EPOCHS = 100
BATCH_SIZE = 2
CLOCK=False
```
To run the traing activate the environment, go to the respective directory and run the proto.py script. If nothing is specified the training will run on a batch size of 4 using no gpu and does not how long a forward pass takes.
However these things can be specified dircelty in the command line in the following way:

```console
(killingplants) usr@dev:~/noplants$ python proto.py [gpu] [batch_size] [value for batch_size] [clock]
```
On The Big Machine we recommend the following setup:
```console
(killingplants) usr@dev:~/noplants$ python proto.py gpu batch_size 24 # higher batch size will exhaust the gpu memory
```
Don't worry if there are errors about gpu stuff in the beginning that is normal.

## Aggregator
The aggregator handles tracking the loss and saving plots. We use a running average of the loss to smoothen the plots. Nothing needs to be adjusted here

## Testing
In order to test a model please adjust the respective prameters in the hyperparameters.py:

``` python
# Testing
DATA_TEST = 'stem_data_cropped_container/stem_data_cropped' # size of pictures doesnt matter, original data usable as well
DATA_TEST_LBL = 'stem_lbl_cropped_container/stem_lbl_cropped'
BATCH_SIZE_TEST = 1
NUM_TESTS = 1
MODEL_TEST_DIR = "models/..." #path tho the model
```
Then run the test.py script. Include gpu if you are running cuda.
```console
(killingplants) usr@dev:~/noplants$ python test.py [gpu]
```

## Model
We build a simple Dense Net from scratch using tensorflow 2.0 as specified by the following paper:https://arxiv.org/pdf/1608.06993.pdf

The Network gets two input, target pairs, one for stem data and one for segmentation data. The first few layers are shared, then the architecture splits in two and there are two readout layers, one for stem and one for segmentation prediction.
