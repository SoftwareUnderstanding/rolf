# Statistics MSs Project: Audio Source Separation
Statistics MSc Project (2020): Audio Source Separation

Some of the results can be seen and listened to on the github page:
<center> https://samargt.github.io/AudioSourceSep/ </center>

## Requirements
- tensorflow 2.2.0 
- tensorflow-addons 0.10.0
- tensorflow-datasets 2.1.0
- tensorflow-probability 0.9.0
- librosa 0.7.2
- pandas 1.1.3
- numpy 1.16.2

## Data

We used mixture of Piano and Violin.

To Transform wav files into melspectrograms, use  wav_to_spec.py from the datasets module.
```bash
python datasets/wav_to_spec.py INPUT_PATH OUTPUT_PATH --use_dB --tfrecords
```

You will need to save the melspectrograms as tfrecords files and **organize them into train/ and test/ folders** in order to train the Generative Models. The exact organization of the dataset should be:

<pre>
DATASET_FOLDER/
|-- source1/
|   |-- train/
|   |    |- source1_train1.tfrecord
|   |    |- source1_train2.tfrecord
|   |    |- ... 
|   |-- test/
|         |- source1_test1.tfrecord
|         |- source2_test2.tfrecord
|         |- ... 
|
|-- source2/
|   |-- train/
|   |-- test/
|
|-- source3/
</pre>

To train a generative model on source1 for instance, one should give the following path: DATASET_FOLDER/source1

## Running Experiments

### train_ncsn.py
Script to train NCSN model
```bash
python train_ncsn.py --dataset [dirpath] --config configs/melspec_ncsnv1.yml

```
The model an tensorboard logs are saved automatically into trained_ncsn/ unless specified otherwise by the --output parameter.

### train_glow.py
Script to train Glow model
```bash
python train_glow.py --dataset [dirpath] --config configs/melspec_glow.yml
```
The model an tensorboard logs are saved automatically into trained_flows/ unless specified otherwise by the --output parameter


### train_noisy_glow.py
Script to fine-tune trained glow model at different noise levels for the BASIS Algorithm
```bash
python train_noisy_glow.py RESTORE_PATH --dataset [dirpath] --config configs/melspec_noisy_glow.yml
```
The model an tensorboard logs are saved automatically into noise_conditioned_flows/ unless specified otherwise by the --output parameter


### run_basis_sep.py
Script to run the BASIS algorithm on the MNIST, CIFAR10 dataset or MelSpectrograms.
Example to separate a mixture of piano and violin:
```bash
python run_basis_sep.py RESTORE_PATH_PIANO RESTORE_PATH_VIOLIN --song_dir [PATH] --model ncsn --config configs/melspec_ncsnv1.yml --output [DIRPATH]

```
The path for song_dir should contain 3 files: mix.wav, violin.wav and piano.wav
RESTORE_PATH_PIANO and RESTORE_PATH_VIOLIN are the path to the piano and violin model checkpoints respectively.

### melspec_inversion_basis.py
Script to inverse the MelSpectrograms from BASIS back to the time domain.
```bash
python melspec_inversion_basis.py DIRPATH --wiener_filter

```
DIRPATH is the path of the folder containing "results.npz", the result from running run_basis_sep.py

## Miscellaneous

- **train_realnvp.py**: Script to train the Real NVP model on MNIST
- **train_utils.py**: Utility functions for training
- **oracle_systems.py**: Oracle Systems for Source Separation (IBM, IRM, MWF). The Code is taken from https://github.com/sigsep/sigsep-mus-oracle and adapted to fit our use.
- **bss_eval_v4.py**: Evaluation of the Separation. Code Taken from https://github.com/sigsep/bsseval and adapted to fit our use.
- **unittest_flow_models.py**: Test the normalizing flows implementation
- **unittest_pipeline.py**: Test the pipeline module
- **technique1_ncsnv2.py**: Compute sigma1 according to technique 1 in http://arxiv.org/abs/2006.09011
- **technique2and4_ncsnv2.py**: Compute num_classes and epsilon according to techniques 2 and 4 in http://arxiv.org/abs/2006.09011

## Modules

### datasets module
#### preprocessing.py
Set of functions to:
- load wav files and spectrograms into tensorflow dataset
- Compute and Save spectrograms from raw audio
- Save dataset as TFRecords and Load TFRecords as dataset

#### dataloader.py
Set of functions to load datasets ready for training or for separation.

#### wav_to_spec.py
Script to convert raw audio (wav files) into Melspectrograms.

### flow_models module
Implement Normalizing flow models.

- **flow_builder** : build flow using Transformed Distribution from Tensorflow-probability
- **flow_glow.py** : implementation of the Glow model
- **flow_realnvp.py** implementaion of the Real NVP model https://arxiv.org/abs/1605.08803
- **flow_tfp_bijectors.py** contains basic bijectors used in complex models
- **flow_tfk_layers.py** contains tf.keras.layers.Layer used for the affine coupling layers. Contains also bijectors implemented with keras (used to compare performances with the tfp implementation)
- **utils.py** : functions such as print_summary to print the trainable variables of the flow models implemented above.
- **flow_pp.py**: Implementation of the Flow++ model (not tested) https://arxiv.org/abs/1902.00275
- flow_tfk_models.py (deprecated) contains a keras Model class used to build a bijector from the bijectors implemented in flow_tfk_layers.py

### ncsn module
Implementation of the Score Network and the Langevin Dynamics to generate samples.
Code taken from https://github.com/ermongroup/ncsn and https://github.com/ermongroup/ncsnv2 and adapted to Tensorflow 2

## References
This work is inspired by 3 main articles: the Glow model, the NCSN model and the BASIS algorithm


#### Glow
```bib
@inproceedings{kingma2018glow,
  title={Glow: Generative flow with invertible 1x1 convolutions},
  author={Kingma, Durk P and Dhariwal, Prafulla},
  booktitle={Advances in neural information processing systems},
  pages={10215--10224},
  year={2018}
}
```

#### NCSN
```bib
@inproceedings{song2019generative,
  title={Generative Modeling by Estimating Gradients of the Data Distribution},
  author={Song, Yang and Ermon, Stefano},
  booktitle={Advances in Neural Information Processing Systems},
  pages={11895--11907},
  year={2019}
}
```

#### BASIS
```bib
@article{jayaram2020source,
  title={Source Separation with Deep Generative Priors},
  author={Jayaram, Vivek and Thickstun, John},
  journal={arXiv preprint arXiv:2002.07942},
  year={2020}
}
```



