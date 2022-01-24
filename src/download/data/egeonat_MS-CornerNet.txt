# MS-CornerNet

This is a pytorch implementation of MS-CornerNet, a Multispectral extension of the CornerNet paper(https://arxiv.org/abs/1808.01244) to work on RGB+T (Thermal) inputs, specifically the kaist dataset. This repository is extended from the base code at: https://github.com/zzzxxxttt/pytorch_simple_CornerNet and the matlab testing code is taken from https://github.com/Li-Chengyang/MSDS-RCNN with slight modifications.
 
## Requirements:
- python>=3.5
- pytorch==0.4.1 or 1.1.0(DistributedDataParallel training only available using 1.1.0)
- tensorboardX(optional)

## Getting Started
1. Disable cudnn batch normalization.
Open `torch/nn/functional.py` and find the line with `torch.batch_norm` and replace the `torch.backends.cudnn.enabled` with `False`.

2. Clone this repo:

3. Compile corner pooling.
    If you are using pytorch 0.4.1, rename ```$MS_CORNERNET_ROOT/lib/cpool_old``` to ```$MS_CORNERNET_ROOT/lib/cpool```, otherwise rename ```$MS_CORNERNET_ROOT/lib/cpool_new``` to ```$MS_CORNERNET_ROOT/lib/cpool```.
    ```
    cd $CornerNet_ROOT/lib/cpool
    python setup.py install --user
    ```

4. Compile NMS.
    ```
    cd $MS_CORNERNET_ROOT/lib/nms
    make
    ```

5. For KAIST training, Download KAIST dataset and put data into ```$CornerNet_ROOT/data/kaist/images``` and ```$CornerNet_ROOT/data/kaist/annotations```. Annotations should then be further separated into two directories ```train_sanitized``` and ```test_improved```

## Train 
### KAIST

#### multi GPU using nn.parallel.DistributedDataParallel
```
python -m torch.distributed.launch --nproc_per_node NUM_GPUS train.py \
        --log_name kaist_hg \
        --dataset kaist \
        --arch large_hourglass \
        --lr 5e-4 \
        --lr_step 90,120 \
        --batch_size 8 \
        --num_epochs 100 \
        --num_workers 1
```

## Evaluate
### COCO
```
python test.py --log_name kaist_hg \
               --dataset kaist \
               --arch large_hourglass
