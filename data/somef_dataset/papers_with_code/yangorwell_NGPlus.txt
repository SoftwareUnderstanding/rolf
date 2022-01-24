## NG+ 

NG+ is a multi-step matrix-product natural gradient method for deep learning. 

## Implementation of NG+ for ImageNet1K with PyTorch

This is an example of training ResNet-50 V1.5 on the ImageNet1K  (ILSVRC2012) dataset. NG+ can finish the training within **40 epochs to top-1 accuracy of 75.9% using 16 Tesla V100 GPUs with batch size 4,096**.

## Model Architecture

The overall network architecture of ResNet-50 is shown below: [link](https://arxiv.org/pdf/1512.03385.pdf)

## Environment Requirements

python 3.8, pytorch (>= 1.5.0), tensorboardX, [NVIDIA DALI](https://developer.nvidia.com/dali), CUDA, CUDNN, and NCCL.

## Quick Start

For batch size 4096 (256 x 16), run the following code:

```python
python -m torch.distributed.launch --master_port 12226 --nproc_per_node=16  main.py --fp16 --batch_size 256  --lr-decay-rate 0.75 --damping 0.35 --lr_init 3.8  --method 'poly' --epoch_end 60 --lr_exponent 6  --warmup_epoch 5 --curvature_momentum 0.9 --datadir /mnt/ILSVRC2012 --logdir your_log_file --decay_epochs 37 --inv-update-freq 1000 --cov-update-freq 1000
```

## Code Structure

- NGPlus.py: Our NG+ optimizer.
- dali_pipe.py: A wrapper over DALI.
- data_manager.py: Utilities about loading datasets.
- logging_utils.py: Utilities about logging.
- main.py: The main script for training.
- nvidia_dali_utils2.py: Utilities about DALI.
- resnet_ngplus.py: The definition of resnet50 model.
- utils.py: Miscellaneous utilities.

## Contact 

We hope that the package is useful for your application. If you have any bug reports or comments, please feel free to email one of the toolbox authors:

- Minghan Yang, yangminghan at pku.edu.cn
- Dong Xu, taroxd at pku.edu.cn
- Zaiwen Wen, wenzw at pku.edu.cn

## Reference

Minghan Yang, Dong Xu, Qiwen Cui, Zaiwen Wen, Pengxiang Xu, NG+ : A Multi-Step Matrix-Product Natural Gradient Method for Deep Learning,  https://arxiv.org/abs/2106.07454

