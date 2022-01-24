# Resource-efficient DNNs for Keyword Spotting using Neural Architecture Search and Quantization [[arXiv]](https://arxiv.org/abs/2012.10138)

```bash
@misc{peter2020resourceefficient,
      title={Resource-efficient DNNs for Keyword Spotting using Neural Architecture Search and Quantization}, 
      author={David Peter and Wolfgang Roth and Franz Pernkopf},
      year={2020},
      eprint={2012.10138},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
}
```

This project uses [ProxylessNAS](https://github.com/mit-han-lab/proxylessnas) to search for resource efficient CNNs for Keyword Spotting on the [Google Speech Commands](https://arxiv.org/abs/1804.03209)
dataset. For further information on ProxylessNAS please refer to the following resources: [[git]](https://github.com/mit-han-lab/proxylessnas) [[arXiv]](https://arxiv.org/abs/1812.00332) [[Poster]](https://file.lzhu.me/projects/proxylessNAS/figures/ProxylessNAS_iclr_poster_final.pdf)

## Contents

  - [Getting Started](#getting-started)
  - [Running the code](#running-your-code)
  - [Authors](#authors)


## Getting Started

These instructions should get you a copy of the project up and running on
your local machine for testing purposes.

### Prerequisites

You need [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for installing python. Install miniconda via

    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
    chmod +x Miniconda-latest-Linux-x86_64.sh
    ./Miniconda-latest-Linux-x86_64.sh

### Environment

To setup the environment create a virtual environment from the included environment.yml

    conda env create -f environment.yml

Activate the environment

    conda activate nas-for-kws

Set the python path

    export PYTHONPATH="/path/to/this/project/"
    
### Data

The [Google Speech Commands](https://arxiv.org/abs/1804.03209) is downloaded and extracted using

    cd /path/to/this/project/src/data/
    python get_speech_commands.py

## Running the code

A trained model is obtained by (1) performing NAS to obtain a good model, and then (2) training the model until convergence.

### Efficient Architecture Search

Set BETA, OMEGA and the output path accordingly and run

    python arch_search.py --path "output_path/" --dataset "speech_commands" --init_lr 0.2 --train_batch_size 100 --test_batch_size 100 --target_hardware "flops" --flops_ref_value 20e6 --n_worker 4 --arch_lr 4e-3 --grad_reg_loss_alpha 1 --grad_reg_loss_beta BETA --weight_bits 8 --width_mult OMEGA --n_mfcc 10
    python run_exp.py --path "output_path/learned_net" --train

### Weight quantization

Weight quantization as a post-processing step by rounding parameters of a trained network is performed using
    
    python run_exp.py --path "output_path/learned_net" --quantize

Quantization aware training using the STE is performed by first changing "num_bits" of all layers in the "net.config" to the desired bit-width and then running

    python run_exp.py --path "output_path/learned_net" --train

### Varying Number of MFCC Features

Set BETA and N_MFCC accordingly and run

    python arch_search.py --path "output_path/" --dataset "speech_commands" --init_lr 0.2 --train_batch_size 100 --test_batch_size 100 --target_hardware "flops" --flops_ref_value 20e6 --n_worker 4 --arch_lr 4e-3 --grad_reg_loss_alpha 1 --grad_reg_loss_beta BETA --weight_bits 8 --width_mult 1 --n_mfcc N_MFCC
    python run_exp.py --path "output_path/learned_net" --train

## Authors

  - **Han Cai and Ligeng Zhu and Song Han** - Original code authors
  - **David Peter** - Updated code to perform NAS for KWS

<p><small>Template folder structure based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small><br>
<small>Readme based on <a target="_blank" href="https://github.com/PurpleBooth/a-good-readme-template">purple booths readme template</a>.</small></p>
