### DetX-FCOS

Implementation of FCOS in PyTorch. <br>
FCOS: Fully Convolutional One-Stage Object Detection. <br>
https://arxiv.org/abs/1904.01355

#### Performance

| mAP(This)~700px | mAP(Paper)-700px | Download | MD5                            |
| :-------------- | ---------------- | ----------------------------------- | :---------------------------------- |
| **37.5%**       | 37.3%            | [Baidu](https://pan.baidu.com/s/1ZkP3_pd3d40InnOAMnA5HA):n1v6 |749c7c972eb2f56cf4ab1d8a61b34c99|

```txt
iters: 87960  epoches: 12
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.375
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.559
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.403
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.206
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.415
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.495
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.310
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.498
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.536
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.316
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.590
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.677
```

#### Usage

```txt
1. Install (PyTorch >= 1.0.0)
sh install.sh

2. Training COCO 1x
python tools/train.py --cfg configs/fcos_r50_sq1025_1x.yaml

3. COCO Eval
copy fcos_r50_sq1025_1x.pkl to weights/
python tools/eval_mscoco.py --cfg configs/fcos_r50_sq1025_1x.yaml

4. Demo
python tools/demo.py --cfg configs/fcos_r50_sq1025_1x.yaml
```
