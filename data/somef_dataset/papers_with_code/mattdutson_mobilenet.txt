# mobilenet

TensorFlow 2 implementation of [MobileNet](https://arxiv.org/abs/1704.04861).

## Data Download

Download the training and validation `.tar` files from [Academic Torrents](https://academictorrents.com/collection/imagenet-2012). Note that ImageNet *does* permit peer-to-peer distribution of the data provided all parties agree to their terms and conditions (see http://image-net.org/download-faq). Academic Torrents will require you to check a box indicating that you agree to the ImageNet terms and conditions before it allows you to download.

Place the downloaded `.tar` files in `~/tensorflow_datasets/downloads/manual` (see the [TensorFlow Datasets documentation](https://www.tensorflow.org/datasets/catalog/imagenet2012)).

## Conda Environment

To create the `mobilenet` environment, run:
```
conda env create -f conda/environment.yml
```
`environment.yml` lists all required Conda and pip packages.

To enable GPU acceleration, instead run:
```
conda env create -f conda/environment_gpu.yml
```
This requires that NVIDIA drivers and CUDA 10.1 be installed (see the [TensorFlow GPU guide](https://www.tensorflow.org/install/gpu)).

After creating one of the above environments, activate it with `conda activate mobilenet`.

## Python API

The Python API is defined in the `mobilenet` package and contains two functions: `dataset.imagenet` and `model.mobilenet`. Details of arguments and outputs are described in the docstrings.

Example usage of `dataset.imagenet`:
```python
from mobilenet.dataset import imagenet

# "data" is a tf.data.Dataset, "n_batches" is an integer
data, n_batches = imagenet(
    'train',  # Can be 'train', 'test', or 'val'
    size=(320, 320),
    augment=True)
```

Example usage of `model.mobilenet`:
```python
from mobilenet.model import mobilenet

# "model" is a tf.keras.Sequential model
model = mobilenet(input_size=(320, 320), l2_decay=1e-3)
```

## Command-Line Interface

The `scripts` subdirectory contains scripts for training (`train.py`), evaluating performance (`test.py`), and saving example input images (`examples.py`). These scripts package the `mobilenet` Python API into a convenient user interface. The full list of options to each script can be viewed with `-h/--help`. For example:
```
./scripts/train.py -h
```
Note that the working directory is the parent directory of `scripts`; this is required in order for Python to find the `mobilenet` package.

`train.py` has a number of options for training hyperparameters (learning rate, optimizer, data augmentation...). The default values were chosen experimentally and should give reasonably high accuracy. That said, feel free to tune them as needed!

## Commit Guidelines

Prepend a flag, followed by the `|` symbol, to each commit indicating the type of change. Possible flag values include:

 - `A` Addition of new features
 - `D` Documentation
 - `E` Change to experiment scripts
 - `F` Bug fixes
 - `M` Miscellaneous or minor modifications
 - `R` Refactoring or restructuring
 - `S` Style or formatting improvements
 - `T` Changes to unit tests
 - `U` Update to dependencies

If there are multiple applicable flags, separate them with commas, for example
`T,R |`.

As a general rule, don't commit experiment scripts until the experiment is completed.
