# Multi-Task Meta-Learning Modification with Stochastic Approximation

This repository contains the code for the paper<br/> 
[**"Multi-Task Meta-Learning Modification with Stochastic Approximation"**](https://arxiv.org/abs/2110.13188).

![Method pipeline](./mtm_pipeline.png)

## Dependencies
This code has been tested on Ubuntu 16.04 with Python 3.8 and PyTorch 1.8.

To install the required dependencies:
```
pip install -r requirements.txt
```

## Usage
To reproduce the results on benchmarks described in our article, use the following scripts. To vary types of the experiments, change the parameters of the scripts responsible for benchmark dataset, shot and way (e.g. miniImageNet 1-shot 5-way or CIFAR-FS 5-shot 2-way).

### MAML
Multi-task modification (MTM) for Model-Agnostic Meta-Learning (MAML) ([Finn et al., 2017](https://arxiv.org/abs/1703.03400)).

Multi-task modifications for MAML are trained on top of baseline MAML model which has to be trained beforehand.

To train **MAML (reproduced) on miniImageNet 1-shot 2-way** benchmark, run:
```
python maml/train.py ./datasets/ \
    --run-name reproduced-miniimagenet \
    --dataset miniimagenet \
    --num-ways 2 \
    --num-shots 1 \
    --num-steps 5 \
    --num-epochs 300 \
    --use-cuda \
    --output-folder ./results
```
To train **MAML MTM SPSA-Track on miniImageNet 1-shot 2-way** benchmark, run:
```
python maml/train.py ./datasets/ \
    --run-name mini-imagenet-mtm-spsa-track \
    --load "./results/reproduced-miniimagenet/model.th" \
    --dataset miniimagenet \
    --num-ways 2 \
    --num-shots 1 \
    --num-steps 5 \
    --task-weighting spsa-track \
    --normalize-spsa-weights-after 100 \
    --num-epochs 40 \
    --use-cuda \
    --output-folder ./results
```
To train **MAML (reproduced) on tieredImageNet 1-shot 2-way** benchmark, run:
```
python maml/train.py ./datasets/ \
    --run-name reproduced-tieredimagenet \
    --dataset tieredimagenet \
    --num-ways 2 \
    --num-shots 1 \
    --num-steps 5 \
    --num-epochs 300 \
    --use-cuda \
    --output-folder ./results
```
To train **MAML MTM SPSA on tieredImageNet 1-shot 2-way** benchmark, run:
```
python maml/train.py ./datasets/ \
    --run-name tiered-imagenet-mtm-spsa \
    --load "./results/reproduced-tieredimagenet/model.th" \
    --dataset tieredimagenet \
    --num-ways 2 \
    --num-shots 1 \
    --num-steps 5 \
    --task-weighting spsa-delta \
    --normalize-spsa-weights-after 100 \
    --num-epochs 40 \
    --use-cuda \
    --output-folder ./results
```
To train **MAML (reproduced) on FC100 5-shot 5-way** benchmark, run:
```
python maml/train.py ./datasets/ \
    --run-name reproduced-fc100 \
    --dataset fc100 \
    --num-ways 5 \
    --num-shots 5 \
    --num-steps 5 \
    --num-epochs 300 \
    --use-cuda \
    --output-folder ./results
```
To train **MAML MTM SPSA-Coarse on FC100 5-shot 5-way** benchmark, run:
```
python maml/train.py ./datasets/ \
    --run-name fc100-mtm-spsa-coarse \
    --load "./results/reproduced-fc100/model.th" \
    --dataset fc100 \
    --num-ways 5 \
    --num-shots 5 \
    --num-steps 5 \
    --task-weighting spsa-per-coarse-class \
    --num-epochs 40 \
    --use-cuda \
    --output-folder ./results
```
To train **MAML (reproduced) on CIFAR-FS 1-shot 5-way** benchmark, run:
```
python maml/train.py ./datasets/ \
    --run-name reproduced-cifar \
    --dataset cifarfs \
    --num-ways 5 \
    --num-shots 1 \
    --num-steps 5 \
    --num-epochs 600 \
    --use-cuda \
    --output-folder ./results
```
To train **MAML MTM Inner First-Order on CIFAR-FS 1-shot 5-way** benchmark, run:
```
python maml/train.py ./datasets/ \
    --run-name cifar-mtm-inner-first-order \
    --load "./results/reproduced-cifar/model.th" \
    --dataset cifarfs \
    --num-ways 5 \
    --num-shots 1 \
    --num-steps 5 \
    --task-weighting gradient-novel-loss \
    --use-inner-optimizer \
    --num-epochs 40 \
    --use-cuda \
    --output-folder ./results
```
To train **MAML MTM Backprop on CIFAR-FS 1-shot 5-way** benchmark, run:
```
python maml/train.py ./datasets/ \
    --run-name cifar-mtm-backprop \
    --load "./results/reproduced-cifar-5shot-5way/model.th" \
    --dataset cifarfs \
    --num-ways 5 \
    --num-shots 1 \
    --num-steps 5 \
    --task-weighting gradient-novel-loss \
    --num-epochs 40 \
    --use-cuda \
    --output-folder ./results
```
To test any of the above-described benchmarks, run:
```
python maml/test.py ./results/path-to-config/config.json --num-steps 10 --use-cuda
```

For instance, to test **MAML MTM SPSA-Track on miniImageNet 1-shot 2-way** benchmark, run:
```
python maml/test.py ./results/mini-imagenet-mtm-spsa-track/config.json --num-steps 10 --use-cuda
```

### Prototypical Networks
Multi-task modification (MTM) for Prototypical Networks (ProtoNet) ([Snell et al., 2017](https://arxiv.org/abs/1703.05175)).

To train **ProtoNet MTM (2 tasks) SPSA-Track with ResNet-12 backbone on miniImageNet 1-shot 5-way** benchmark, run:
```
python protonet/train.py \
    --dataset miniImageNet \
    --network ResNet12 \
    --tracking \
    --task-number 2 \
    --train-shot 1 \
    --train-way 5 \
    --val-shot 1 \
    --val-way 5
```
To test **ProtoNet MTM SPSA-Track with ResNet-12 backbone on miniImageNet 1-shot 5-way** benchmark, run:
```
python protonet/test.py --dataset miniImageNet --network ResNet12 --shot 1 --way 5
```
To train **ProtoNet MTM Backprop with 64-64-64-64 backbone on CIFAR-FS 1-shot 2-way** benchmark, run:
```
python protonet/train.py \
    --dataset CIFAR_FS \
    --train-weights \
    --train-weights-layer \
    --train-shot 1 \
    --train-way 2 \
    --val-shot 1 \
    --val-way 2
```
To test **ProtoNet MTM Backprop with 64-64-64-64 backbone on CIFAR-FS 1-shot 5-way** benchmark, run:
```
python protonet/test.py --dataset CIFAR_FS --shot 1 --way 2
```
To train **ProtoNet MTM Inner First-Order with 64-64-64-64 backbone on FC100 10-shot 5-way** benchmark, run:
```
python protonet/train.py \
    --dataset FC100 \
    --train-weights \
    --train-weights-opt \
    --train-shot 10 \
    --train-way 5 \
    --val-shot 10 \
    --val-way 5
```
To test **ProtoNet MTM Inner First-Order with 64-64-64-64 backbone on FC100 10-shot 5-way** benchmark, run:
```
python protonet/test.py --dataset FC100 --shot 10 --way 5
```
To train **ProtoNet MTM SPSA with 64-64-64-64 backbone on tieredImageNet 5-shot 2-way** benchmark, run:
```
python protonet/train.py \
    --dataset tieredImageNet \
    --train-shot 5 \
    --train-way 2 \
    --val-shot 5 \
    --val-way 2
```
To test **ProtoNet MTM SPSA with 64-64-64-64 backbone on tieredImageNet 5-shot 2-way** benchmark, run:
```
python protonet/test.py --dataset tieredImageNet --shot 5 --way 2
```

## Acknowledgments

Our code uses some dataloaders from [**Torchmeta**](https://github.com/tristandeleu/pytorch-meta).

Code in maml folder is based on the extended implementation from [**Torchmeta**](https://github.com/tristandeleu/pytorch-meta) and [**pytorch-maml**](https://github.com/tristandeleu/pytorch-maml). The code has been updated so that baseline scores more closely follow those of the original MAML paper.

Code in protonet folder is based on the implementation from [**MetaOptNet**](https://github.com/kjunelee/MetaOptNet). All .py files in this folder except for dataloaders.py and optimize.py were adopted from this implementation and modified afterwards. A copy of Apache License, Version 2.0 is available in protonet folder.

## Citation

If you want to cite our paper, you can use the following BibTeX entry:
```
@article{boiarov2021multi,
  title={Multi-Task Meta-Learning Modification with Stochastic Approximation},
  author={Boiarov, Andrei and Khabarlak, Konstantin and Yastrebov, Igor},
  journal={arXiv preprint arXiv:2110.13188},
  year={2021}
}
```
