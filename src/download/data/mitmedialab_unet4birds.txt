# Wetland Avian Source Separation (WASS)

The repository is currently under **development**.

## Install

To install all the necessary dependencies for this repository install all the python libraries using the `pip` command *(may require the use of sudo)*:
```bash
$ pip3 install -r requirements.txt
```

```txt
torch>=1.2.0
numpy>=1.13.3
torchaudio>=0.3.0
matplotlib>=2.1.1
tqdm>=4.36.1
PyYAML>=5.3
```

## Usage

Scripts must be used with the following command:
```bash
# Help
$ python3 -m wass [-h] 

# Train
$ pyfhon3 -m wass [-t] [--config CONFIG] [--gpu GPU [GPU ...]]

# Demo
$ python3 -m wass [-d] [--model MODEL] [--mixture MIXTURE] [--dest DEST]
```

```bash
$ python3 -m wass -h
usage: wass [-h] [-t] [--config CONFIG] [--gpu GPU [GPU ...]] [-d]
            [--model MODEL] [--mixture MIXTURE] [--dest DEST]

Wetland Avian Source Separation (WASS) -- Scripts
	Source Code:
		https://github.com/mitmedialab/WetlandAvianSourceSeparation
	Authors:
		https://github.com/FelixMichaud
		https://github.com/yliess86
		https://github.com/slash6475
    

optional arguments:
  -h, --help           show this help message and exit

  -t, --train          training
  --config CONFIG      training config file path
  --gpu GPU [GPU ...]  cuda devices

  -d, --demo           demonstration
  --model MODEL        path to a trained model
  --mixture MIXTURE    path to a mixture audio file
  --dest DEST          path to save separated sources
```

### Dataset

The Dataset is a simple class inheriting from `torch.util.data.Dataset`. It provides access to its length and items through a pythonic API. The class also offers ways to generate the Dataset, either from a configuration file *(see `Configuration` section)* or from default values *(see code)*.

```python
from wass.dataset import CompositionDataset


# Generate dataset from config
train_dataset = CompositionDataset.generate_from_config(
    "dataset/train",
    config=config,
    size=20000
)

# Get length
print(len(train_dataset))   # >>> 20000

# Get datum
#   composition: [B, 1, S]
#       composition audio 
#   sequences: [B, C, S]
#       individual sequences from the composition
composition, sequences = train_dataset[0]
```

#### Data

The Dataset is generated out of birds and ambient samples. Those samples are 
available in the data folder and contains ten bird species:

```
data/
|   ambient/
|   |   airplane/
|   |   rain/
|   |   wind/
|   birds/
|   |   blue_jay/
|   |   cardinal/
|   |   crow/
|   |   eastern_wood_pewee/
|   |   flicker/
|   |   green_heron/
|   |   purple_finch/
|   |   veery/
|   |   willow_flycatcher/
|   |   yellow_vireo/
```

#### Configuration

The generated compositions can be controlled with various parameters through a configuration file saved as a `.yaml` file. The configuration file allows two type of training. You can either decide to separate all labels among all labels or specific labels among all labels.

```python
from wass.audio.dataset import ComposerConfig


# Create configuration
config = ComposerConfig(
    "data/birds",
    "data/ambient",
    label_size=(0, 8),         # Range of samples for birds
    ambient_size=(0, 2),       # Range of samples for ambient
    label_scale=(0.5, 1.5),    # Range of volume scale for birds
    ambient_scale=(0.4, 1.4),  # Range of volume scale for ambient
    duration=4,                # Duration in second
    sr=16000,                  # Sample rate
    snr=(0, 100),              # Signal to noise ratio for Noise
    focus=None                 # Ability to extract list of labels only
)

# Save configuration
config.save("path.yaml")

# Retrieve from file
config = ComposerConfig.load("path.yaml)
```

Composer configuration files are provided in the `config/composer` folder. Here is an example of the default configuration file `default.yaml`:

```yaml
label_directory: data/birds
ambient_directory: data/ambient
label_size: !!python/tuple [0, 8]
ambient_size: !!python/tuple [0, 2]
label_scale: !!python/tuple [0.5, 1.5]
ambient_scale: !!python/tuple [0.4, 1.4]
duration: 4
sr: 16000
snr: !!python/tuple [0, 100]
```

### Training

Training is performed through the use of the `Solver` class. The solver takes care of every part of the training, including:
- Creating the Dataset
- Initializing the Model, Criterion, and Optimizer
- Training and Testing Procedure
- Saving Progress

This is performed using configuration files, one for training information *(see below in the `Configuration` section)* that itself refers to the second concerning the Dataset Composer configuration *(see above in the `Dataset` Section for this one)*. This choice allows for great flexibility for running different experiments with different parameters.

```python
from wass.train import Solver
from wass.train import TrainingConfig


# Load Training Config file and check parameters
config = TrainingConfig.load("config/training/default.yaml")
print(config.__dict__, "\n")

# Initialize Solver with config, cuda support and run
solver = Solver(config, cuda=True)
solver()
```

#### Configuration

As mentioned above, the solver uses the configuration in the same way the dataset generation is handled. The training configuration file contains all hyperparameter fields for training. Mapping with different `ComposerConfig` files is enabled for modularity.

```python
from wass.train import TrainingConfig


# Create Configuration
config = TrainingConfig(
    epochs=10,                        # number of epoch
    lr=1e-3,                          # learning rate for the optimizer
    max_norm=5,                       # clip gradient norm
    batch_size=16,                    # batch size
    n_workers=4,                      # number of worker for dataloader
    n_train=20000,                    # number of training samples
    n_test=4000,                      # number of testing samples
    composer_conf_path=composer_conf, # path to a composer configuration
    saving_path="results",            # path to save the experiment
    exp_name="experiment_01",         # experiment name
    saving_rate=2,                    # rate to save progress
    pit=False                         # use pit training
)
```

Training configuration files are provided in the `config/training` folder. Here is an example of the default configuration file `default.yaml` mapped with the default `ComposerConfig`:

```yaml
epochs: 100
lr: 1e-3
max_norm: 5
batch_size: 8
n_workers: 2
n_train: 200
n_test: 40
composer_conf_path: config/composer/default.yaml
saving_path: results
exp_name: default
saving_rate: 2
pit: false
```

### Inference

Inference is part of the repository for demonstration purposes. If you want to use a trained model prefer to export the model as a `torch.script` file using the provided *(TODO)* JIT compile and quantizer tool. This will allow you to use the model outside of the repository scope *(see Optimization section TODO)*.

Neverthless demo script is available for you to try a freshly trained model by providing the model path, the audio mixture path and the destination path to save the results as wave files *(see above ath the begining of usage section)*.


### Optimization

This section is **under development**.

## References

### Papers
- [Conv-TasNet] - TasNet: Surpassing Ideal Time-Frequency Masking for Speech Separation
- [SI SDR] - SDR – Half-Baked or Well Done?

### Implementations
- [Authors] - Github repository
- [Naplab] - Github repository
- [Jhuiac] - Github repository

[Conv-TasNet]: https://arxiv.org/abs/1809.07454
[SI SDR]: https://arxiv.org/pdf/1811.02508.pdf
[Authors]: https://github.com/kaituoxu/Conv-TasNet
[Naplab]: https://github.com/naplab/Conv-TasNet
[Jhuiac]: https://github.com/jhuiac/conv-tas-net

## Authors

- [Felix MICHAUD]
- [Yliess HATI]
- [Clement DUHART]

[Felix MICHAUD]: https://github.com/FelixMichaud
[Yliess HATI]: https://github.com/yliess86
[Clement DUHART]: https://github.com/slash6475
