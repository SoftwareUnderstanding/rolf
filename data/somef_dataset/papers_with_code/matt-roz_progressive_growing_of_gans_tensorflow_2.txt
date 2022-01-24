## Progressive Growing of GANs - TensorFlow 2 Implementation

![Representative image](res/representative_image_512x1792x3.png)

This is a TensorFlow 2 implementation of *Progressive Growing of GANs*. The original implementation was provided by the authors
**Tero Karras** (NVIDIA), **Timo Aila** (NVIDIA), **Samuli Laine** (NVIDIA), **Jaakko Lehtinen** (NVIDIA and Aalto University).
Please cite the original authors and their work (**not** this repository):

[Paper (arXiv)](http://arxiv.org/abs/1710.10196) <br>
[TensorFlow 1 Implementation (github)](https://github.com/tkarras/progressive_growing_of_gans)

---
### Overview
The repository at hand was written to get myself more comfortable and familiar with TensorFlow 2. It aims to provide a maintainable and well-written implementation of Progressive GANs in TensorFlow 2. It follows the best practices for **distributed computing with custom training loops and dynamic models** according to [TensorFlow's API](https://www.tensorflow.org/api_docs/python/). This repository aims to use the *highest level API* available in TensorFlow 2 for each building block (dataset, model, layer, etc.):

* [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset): a `celeb_a_hq` pipeline built via [tensorflow_datasets](https://www.tensorflow.org/datasets)
* [`tf.keras.Model`](https://www.tensorflow.org/api_docs/python/tf/keras/Model): functional API implementations of models (for shape inference at `model.build()` time)
* [`tf.autograph`](https://www.tensorflow.org/api_docs/python/tf/autograph): tracing/compiling python functions for faster graph mode execution 
  * using `tf.function` as a function annotation where appropriate (e.g. [`losses.py`](losses.py)) for static functions
  * using `tf.function` as a function call to manually determine re-tracing of python functions at runtime (necessary to execute dynamic models in graph mode)
* [`tf.keras.layers`](https://www.tensorflow.org/api_docs/python/tf/keras/layers):
  * subclassing [`Layer`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer) for custom layers defined in the original implementation ([`PixelNormalization`](https://github.com/tkarras/progressive_growing_of_gans/blob/master/networks.py#L120), [`StandardDeviationLayer`](https://github.com/tkarras/progressive_growing_of_gans/blob/master/networks.py#L127))
  * subclassing [`Wrapper`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Wrapper) to realize the *weight scaling trick* for any `tf.keras.layers.Layer` as proposed in the original paper
* [`tf.distribute.Strategy`](https://www.tensorflow.org/api_docs/python/tf/distribute/Strategy): allowing the same code base to be run executed with different distribution stratgies **without** code repetition (`DefaultStrategy`, [`MirroredStrategy`](https://www.tensorflow.org/api_docs/python/tf/distribute/MirroredStrategy), [`MultiWorkerMirroredStrategy`](https://www.tensorflow.org/api_docs/python/tf/distribute/experimental/MultiWorkerMirroredStrategy))

The original TensorFlow 1 repository took roughly 2 weeks of traintime for a 1024x1024x3 network on a single V100. This repository takes 5 days, 11hrs for the same network on a Quadro RTX 6000. Here are three 256x256x3 interpolation results:

![Example Gif](res/inter3.gif) ![Example Gif](res/inter2.gif) ![Example Gif](res/inter1.gif)

---
### Differences to the original TensorFlow 1 contribution

This repository, in its default configuration [`config.py`](config.py), differs from its original contribution in the following ways:
* The original contribution linearly increases `alpha` (*image smoothing* scalar for linear interpolation) over 800k images. This repository increases `alpha` linearly over 810k images.
* The original contribution trains each stage > 2 for 1.6M images. This repository trains each stage for 1.62M images.
* The original contribution trains stage 2 for 800k images. This repository trains stage 2 for 1.62M images.
* The original contribution [resets optimizer states](https://github.com/tkarras/progressive_growing_of_gans/blob/master/tfutil.py#L375) after each stage increase. This repository re-initializes its optimizers.
* The original contribution allows for configurable [discriminator repeats](https://github.com/tkarras/progressive_growing_of_gans/blob/master/train.py#L228). This repository has no discriminator repeats.
* The original contribution has an optional [label input](https://github.com/tkarras/progressive_growing_of_gans/blob/master/networks.py#L146) and computes [label loss penalties](https://github.com/tkarras/progressive_growing_of_gans/blob/master/loss.py#L35), if labels are given. This repository doesn't support labels.
* The original contribution [`alpha` smooths (linearly interpolates)](https://github.com/tkarras/progressive_growing_of_gans/blob/master/networks.py#L214) all intermediate image outputs within its models. This repository only [linearly interpolates the image of the last block of the current stage](https://github.com/matt-roz/progressive_growing_of_gans_tensorflow_2/blob/master/model.py#L127).
* The original contribution provides a [recursive](https://github.com/tkarras/progressive_growing_of_gans/blob/master/networks.py#L217) network definition. This repository builds its models linearly. 

I am certain there are more differences, but these are the major ones I could think of. Let me know if you spot any other major discrepancies.

---
### System requirements
* Linux with 64-bit Python 3.6 and `python-pydot`, `python-pydotplus` installed (see installation)
* 16GB system memory and one or more high-end NVIDIA Turing, Pascal or Volta GPUs with 16GB of DRAM. 
* NVIDIA driver 440.64.00 or newer, CUDA toolkit 10.1 or newer, cudNN 7.6.5 or newer
   * Disclaimer: It's likely possible to run this repository on older software installations (specifically if you are willing to run pre tensorflow 2.1.0). If you are going down this road some manual adaptions are likely required :-/

---
### Installation & Training
Personally I use virtualenv, but you can use conda, docker or any other type of virtualenv/containerization technique that floats your boat. Make sure the system requirements mentioned above are met.

Install required packages:

    sudo apt-get install python3-pydot python3-pydotplus python3-venv
    
Clone the repository:

    git clone git@github.com:matt-roz/progressive_growing_of_gans_tensorflow_2.git
    cd progressive_growing_of_gans_tensorflow_2
    
Setup your environment: 
    
    python3 -m venv venv
    source ven/bin/activate
    pip install --upgrade pip setuptools
    pip install -r requirements.txt

Adapt the configuration for your system (specifically `data_dir`, `log_dir` and `out_dir` paths):

    nano config.py

Train Progressive-GANs:

    python main.py
    
You'll see the project logfile and the TensorBoard logfile in `log_dir`; model checkpoints as well as eval images will be stored in `out_dir`. 

---
### Configuration
The following options are configurable via [`config.py`](config.py). This config file is backed up for each run in its respective output directory. By default the configuration will train a 256x256x3 network for CelebA-HQ using a single GPU (index 0 GPU). 

<details><summary>Global Settings</summary>

| identifier | dtype | default | meaning |
|---|---|---|---|
| save | bool | True | de-/activates model saving and checkpointing |
| evaluate | bool | True | de-/activates model  evaluation|
| logging | bool | True | de-/activates file logging (incl. TensorBoard) |
| out_dir | str, os.PathLike | '/media/storage/outs/' | directory for output files (images, models) |
| log_dir | str, os.PathLike | '/media/storage/outs/' | directory for logging (logfile, tensorboard) |
| data_dir | str, os.PathLike | '~/tensorflow_datasets' | directory to load tensorflow_datasets from |
| train_eagerly | bool | False | de-/activates execution of train_step in graph mode |
| XLA | bool | False | de-/activates XLA JIT compilation for train_step |
| strategy | str | 'default' | distribution strategy |
| checkpoint_freq | uint | 54 | epoch frequency to checkpoint models with (0 = disabled) |
| eval_freq | uint | 1 | epoch frequency to evaluate models with (0 = disabled) |
| log_freq | uint | 1 | epoch frequency to log with (0 = disabled) |
| global_seed | int | 1000 | global tensorflow seed |


**Note:** If you want to train on a cluster (strategy = `'multimirrored'`), make sure that your environment variable `$TF_CONFIG` is correctly configured for each node. This repository defines the worker at index 0 as its chief. The chief will handle all file outputs (make sure he has the necessary rights to write within the defined output directories).

</details>

<details><summary>Model</summary>

| identifier | dtype | default | meaning |
|---|---|---|---|
| leaky_alpha | float | 0.2 | leakiness of LeakyReLU activations |
| generator_ema | float | 0.999 | exponential moving average of final_generator |
| resolution | uint | 256 | final resolution |
| noise_dim | uint | 512 | noise_dim generator projects from |
| epsilon | float | 1e-8 | small constant for numerical stability in model layers |
| data_format | str | 'channels_last' | defines order of dimensions for images |
| use_bias | bool | True | de-/activates usage of biases in all trainable layers |
| use_stages | bool | True | de-/activates progressive training of model in stages |
| use_fused_scaling | bool | True | de-/activates up- and downsampling of images via strides=(2, 2) in Conv2D and Conv2DTranspose |
| use_weight_scaling | bool | True | de-/activates weight scaling trick |
| use_alpha_smoothing | bool | True | de-/activates smoothing in an image from a previous block after increasing the model to a new stage |
| use_noise_normalization | bool | True | de-/activates pixel_normalization on noise input at generator start |

</details>

<details><summary>Training</summary>

| identifier | dtype | default | meaning |
|---|---|---|---|
| epochs | uint | 432 | number of epochs to train for |
| epochs_per_stage | uint | 54 | number of epochs per stage |
| alpha_init | float | 0.0 |  initial alpha value to smooth in images from previous block |
| use_epsilon_penalty | bool | True | de-/activates epsilon_drift_penalty applied to discriminator loss |
| drift_epsilon | float | 0.001 |  epsilon scalar for epsilon_drift_penalty |
| use_gradient_penalty | bool | True | de-/activates gradient_penalty applied to discriminator loss |
| wgan_lambda | float | 10.0 | wasserstein lambda scalar for gradient_penalty |
| wgan_target | float | 1.0 | wasserstein target scalar for gradient_penalty |
| random_image_seed | int | 42 | seed for fixed-random evaluate images |

</details>

<details><summary>Data Pipeline</summary>

| identifier | dtype | default | meaning |
|---|---|---|---|
| registered_name | str | 'celeb_a_hq' | name argument for tensorflow_datasets.load |
| split | str  | 'train' | split argument for tensorflow_datasets.load |
| num_examples | uint | 30000 | number of examples train dataset will contain according to loaded split |
| caching | bool | False | de-/activates dataset caching to file or system memory (see cache_file) |
| cache_file | str, os.PathLike | '/tmp/{timestamp}-tf-dataset.cache' | location of temporary cache_file ("" = load entire dataset into system memory) |
| process_func | function | celeb_a_hq_process_func | function to process each dataset entry with |
| map_parallel_calls | int | tf.data.experimental.AUTOTUNE | number of parallel entries to apply 'process_functions' asynchronously |
| prefetch_parallel_calls | int | tf.data.experimental.AUTOTUNE | number of parallel threads to prefetch entries with concurrently |
| replica_batch_sizes | dict | {2: 128, 3: 128, 4: 128, 5: 64, 6: 32, 7: 16, 8: 8, 9: 6, 10: 4}  | per replica batch size at stage |
| buffer_sizes | dict | {2: 5000, 3: 5000, 4: 2500, 5: 1250, 6: 500, 7: 400, 8: 300, 9: 250, 10: 200}   | buffer size at stage |

</details>

<details><summary>Optimizer</summary>

| identifier | dtype | default | meaning |
|---|---|---|---|
| learning_rates | dict | {2: 1e-3, 3: 1e-3, 4: 1e-3, 5: 1e-3, 6: 1e-3, 7: 1e-3, 8: 1e-3, 9: 1e-3, 10: 1e-3} | learning_rate at stage |
| beta1 | float  | 0.0 | exponential decay rate for the 1st moment estimates |
| beta2 | float  | 0.99 | exponential decay rate for the 2nd moment estimates |
| epsilon | float | 1e-8 | small constant for numerical stability |

</details>

<details><summary>Logging</summary>

| identifier | dtype | default | meaning |
|---|---|---|---|
| device_placement | bool | False |  de-/activates TensorFlow device placement logging |
| level | str | 'INFO' | log level of project logger |
| filename | str, os.PathLike | '{timestamp}-{host}-logfile.log'  | name of resulting log file |
| format | str | '%(asctime)s - %(name)s - %(levelname)s - %(message)s' | log formatting for formatter |
| datefmt | str | '%m/%d/%Y %I:%M:%S %p' | datetime formatting for formatter |
| adapt_tf_logger | bool | True | de-/activates overriding of tf_logger configuration |
| tf_level | str | 'ERROR' | log level of TensorFlow logging logger |

</details>

---
### Roadmap
The following features are planned for the near future.

- [ ] add mixed_precision (fp16) training
  - [ ] make models [`model.py`](model.py) dtype aware
- [x] support for NCHW (channel_first) data format
  - [x] make custom layers [`layers.py`](layers.py) data_format aware
  - [x] make models [`model.py`](model.py) data_format aware
  - [x] configurable via [`config.py`](config.py)
- [ ] support for non RGB-images
  - [ ] make custom layers [`layers.py`](layers.py) num_channels aware
  - [ ] make models [`model.py`](model.py) num_channels aware
  - [ ] configurable via [`config.py`](config.py)
- [ ] implement metrics 
  - [ ] MS-SIM
  - [ ] FID
  - [ ] R&R

PRs are very welcome and appreciated!

---
### Personal Note
Located in Germany, passionate about ML and looking for opportunities world wide: `matthiasrozanski[at]gmail[dot]com` 

If you are looking for additions to your (research/engineering) team: Don't hesitate reaching out - I ain't biting :)