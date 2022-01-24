# SSD (Single Shot MultiBox Detector) - Tensorflow 2.0

> tensorflow_datasets will automatically download the VOC dataset

## Preparation
- Install necessary dependencies:
```
pip install -r requirements.txt
```

## Training
Arguments for the training script:

```
>> python train.py --help
usage: train.py [-h] [--data-dir DATA_DIR]
                [--arch ARCH] [--batch-size BATCH_SIZE]
                [--num-batches NUM_BATCHES] [--neg-ratio NEG_RATIO]
                [--initial-lr INITIAL_LR] [--momentum MOMENTUM]
                [--weight-decay WEIGHT_DECAY] [--num-epochs NUM_EPOCHS]
                [--checkpoint-dir CHECKPOINT_DIR]
                [--pretrained-type PRETRAINED_TYPE] [--gpu-id GPU_ID]
```
Arguments explanation:
-  `--data-dir` dataset directory where tfds will store data
-  `--arch` SSD network architecture (ssd300 or ssd512)
-  `--batch-size` training batch size
-  `--num-batches` number of batches to train (`-1`: train all)
-  `--neg-ratio` ratio used in hard negative mining when computing loss
-  `--initial-lr` initial learning rate
-  `--momentum` momentum value for SGD
-  `--weight-decay` weight decay value for SGD
-  `--num-epochs` number of epochs to train
-  `--checkpoint-dir` checkpoint directory
-  `--pretrained-type` pretrained weight type (`base`: using pretrained VGG backbone, other options: see testing section)
-  `--gpu-id` GPU ID

- how to train SSD300 using PASCAL VOC2007 for 100 epochs:

```
python train_tfds.py --num-epochs 100
```

- how to train SSD512 using PASCAL VOC2012 for 120 epochs on GPU 1 with batch size 8 and save weights to `./checkpoints_512`:

```
python train_tfds.py --arch ssd512 --num-epochs 120 --batch-size 8 --checkpoint_dir ./checkpoints_512 --gpu-id 1
```

## Testing
Arguments for the testing script:
```
>> python test.py --help
usage: test.py [-h] [--data-dir DATA_DIR]
               [--arch ARCH] [--num-examples NUM_EXAMPLES]
               [--pretrained-type PRETRAINED_TYPE]
               [--checkpoint-dir CHECKPOINT_DIR]
               [--checkpoint-path CHECKPOINT_PATH] [--gpu-id GPU_ID]
```
Arguments explanation:
-  `--data-dir` dataset directory (must specify to VOCdevkit folder)
-  `--arch` SSD network architecture (ssd300 or ssd512)
-  `--num-examples` number of examples to test (`-1`: test all)
-  `--checkpoint-dir` checkpoint directory
-  `--checkpoint-path` path to a specific checkpoint
-  `--pretrained-type` pretrained weight type (`latest`: automatically look for newest checkpoint in `checkpoint_dir`, `specified`: use the checkpoint specified in `checkpoint_path`)
-  `--gpu-id` GPU ID

- how to test the first training pattern above using the latest checkpoint:

```
python test.py --data-dir ./data --checkpoint_dir ./checkpoints
```

- how to test the second training pattern above using the 100th epoch's checkpoint, using only 40 examples:

```
python test.py --data-dir ./data --arch ssd512 --checkpoint_path ./checkpoints_512/ssd_epoch_100.h5 --num-examples 40
```

## Reference
- Single Shot Multibox Detector paper: [paper](https://arxiv.org/abs/1512.02325)
- Caffe original implementation: [code](https://github.com/weiliu89/caffe/tree/ssd)
- Pytorch implementation: [code] (https://github.com/ChunML/ssd-pytorch)
