# Object Detection

This repository contains my study (maybe **buggy**) code for the following object detection methods:

* FasterRCNN (https://arxiv.org/abs/1506.01497)
* FPN (https://arxiv.org/abs/1612.03144)
* RetinaNet (https://arxiv.org/abs/1708.02002)
* FCOS (https://arxiv.org/abs/1904.01355)

## Environment

1. tensorflow_gpu-1.14
2. tensorpack-0.10.1 (for fast data loading)

## Usage

1. Download the `resnet_resnet-nhwc-2018-02-07/model.ckpt-112603` from Google.
2. Set the related path, GPU configuration, batch_size, ..., in `config.py`
3. To train `FasterRCNN`, set in `config.py`

    ```
    cfg.MODE_FRCNN = True
    cfg.MODE_FPN = False
    cfg.MODE_MASK = False
    cfg.MODE_FCOS = False
    cfg.MODE_RETINANET = False
    ```

    run `python train_FasterRCNN.py train`

4. For evaluation, run `python train_FasterRCNN.py eval`

## Evaluation Results on COCO2017

1. FasterRCNN

    ```
    # coco_val2017 results (20191129) 1x LR-schedule
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.343
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.556
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.361
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.152
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.380
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.493
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.296
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.460
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.479
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.255
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.530
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.666
    ```

2. FPN
    ```
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.358
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.578
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.381
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.209
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.393
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.466
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.305
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.489
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.518
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.339
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.556
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.648
    ```
3. RetinaNet
    ```
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.326
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.501
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.356
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.180
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.364
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.427
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.298
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.485
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.522
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.322
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.567
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.680
    ```
4. FCOS
    ```
    V100 2GPU, bs=8 (focal loss from tensorflow/tpu/retinanet sigmoid_focal_loss4)
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.362
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.560
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.383
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.207
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.396
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.468
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.309
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.502
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.531
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.340
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.574
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.684
    ```

## Tensorboard

![](tensorboard.png)


## Finally
Some codes are copied from `tensorflow/tpu` or `tensorpack`.
Any suggestions and comments are welcomed and greatly appreciated! 

