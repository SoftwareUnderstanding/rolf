# Progressively growing GAN
Tensorflow 2 implementation of Progressive Growing of GANs for Improved Quality, Stability, and Variation: https://arxiv.org/abs/1710.10196
<br>
Code is based on official implementation: https://github.com/tkarras/progressive_growing_of_gans

This implementation allows finer control of a training process and model complexity: 
one can use different parameters which define number of filters of each network (consider function `n_filters()` in `networks.py`), 
size of latent vector, use different activation functions, add/remove biases, set different numbers of images for each stage, 
use different optimizers settings,
etc.


## Model training
To train a model one needs:

1. Define a training config (example in `default_config.json`, all available options and their default values can be found in `utils.py`). *Note:* some values in config are different from original implementation due to memory constraints.
2. Optionally configure gpu memory options (consider **GPU memory usage** section)
3. Optionally set training mode (consider **Training speed** section)
4. Start training with command: <br>
```
python train.py --config=path_to_config (e.g. --config=default_config.json)
```

## Training speed

To get maximum performance one should prefer training each model in a separate process (`single_process_training` in `train.py`), 
as in this case all GPU resources are released after process is finished.  <br>
Another way to incease performance is to use mixed precision training, which not just speeds operations up (especially on Nvidia cards with compute capability 7.0 or higher, e.g. Turing GPUs), but also allows to increase batch size. <br>
*Note:* for now restoring/saving optimizer state is not implemented, so if optimizers states should not be reset a model must be trained in a single process. By default models in official implementation reset oprimizer state for new level of details.

## GPU memory usage

To control GPU memory usage one can refer to a function `prepare_gpu()` in `utils.py`. 
<br>
Depending on your operating system use case you might want to change memory managing. 
By default on Linux memory_growth option is used, while on Windows memory is limited with some reasonable number to allow use of PC (such as opening browsers with small number of tabs).
<br>
*Note:* code was used with GPUs with 8 Gb of memory, so if your card has more/less memory it is strongly recommended to consider modifying `prepare_gpu()` function. 

## System requirements

- Code was tested on Linux and Windows. Though when running on Windows you can face additional warning messages from Tensorflow, also, it takes more time to start training (usually up to several additional minutes for each stage), however, it doesn't affect training process
- Requirements can be installed from `conda_environment.yml`(`conda env create -f conda_environment.yml`)


## Further improvements
- Saving/restoring optimizers states
- Multi GPU support

