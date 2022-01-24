# RegNetx-YOLOv3
PyTorch implementation of YOLOv3 with RegNetx backbone. Models regnetx_002 and regnetx_006 supported.<br>
The YOLOv3 code implementation has been taken from https://github.com/eriklindernoren/PyTorch-YOLOv3 and the RegNetx implementation is from https://github.com/d-li14/regnet.pytorch/blob/master/regnet.py <br>

# Papers
YOLOv3 - https://arxiv.org/abs/1804.02767 <br>
Paper defining RegNetx architecture - Facebook's https://openaccess.thecvf.com/content_CVPR_2020/papers/Radosavovic_Designing_Network_Design_Spaces_CVPR_2020_paper.pdf <br>

# Training
```
python train.py --help
usage: train.py [-h] [--epochs EPOCHS] [--batch_size BATCH_SIZE]
                [--gradient_accumulations GRADIENT_ACCUMULATIONS]
                [--data_config DATA_CONFIG]
                [--pretrained_weights PRETRAINED_WEIGHTS] [--n_cpu N_CPU]
                [--img_size IMG_SIZE]
                [--checkpoint_interval CHECKPOINT_INTERVAL]
                [--evaluation_interval EVALUATION_INTERVAL]
                [--compute_map COMPUTE_MAP]
                [--multiscale_training MULTISCALE_TRAINING]
                [--regnetx_model REGNETX_MODEL]

optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS       number of epochs
  --batch_size BATCH_SIZE
                        size of each image batch
  --gradient_accumulations GRADIENT_ACCUMULATIONS
                        number of gradient accums before step
  --data_config DATA_CONFIG
                        path to data config file
  --pretrained_weights PRETRAINED_WEIGHTS
                        if specified starts from checkpoint model
  --n_cpu N_CPU         number of cpu threads to use during batch generation

  --img_size IMG_SIZE   size of each image dimension
  --checkpoint_interval CHECKPOINT_INTERVAL
                        interval between saving model weights
  --evaluation_interval EVALUATION_INTERVAL
                        interval evaluations on validation set
  --compute_map COMPUTE_MAP
                        if True computes mAP every tenth batch
  --multiscale_training MULTISCALE_TRAINING
                        allow for multi-scale training
  --regnetx_model REGNETX_MODEL
                        the regnet model to be used as backbone
  ```

# Testing
```
python test.py --help
usage: test.py [-h] [--batch_size BATCH_SIZE] [--model_def MODEL_DEF]
               [--data_config DATA_CONFIG] [--class_path CLASS_PATH]
               [--iou_thres IOU_THRES] [--conf_thres CONF_THRES]
               [--nms_thres NMS_THRES] [--n_cpu N_CPU] [--img_size IMG_SIZE]
               [--regnetx_model REGNETX_MODEL]
```

## Credit
### YOLOv3: An Incremental Improvement
_Joseph Redmon, Ali Farhadi_ <br>

[[Paper]](https://pjreddie.com/media/files/papers/YOLOv3.pdf)
```
@article{yolov3,
  title={YOLOv3: An Incremental Improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal = {arXiv},
  year={2018}
}
```
### Designing Network Design Spaces <br>
[[Paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Radosavovic_Designing_Network_Design_Spaces_CVPR_2020_paper.pdf) 

```
@inproceedings{radosavovic2020designing,
  title={Designing network design spaces},
  author={Radosavovic, Ilija and Kosaraju, Raj Prateek and Girshick, Ross and He, Kaiming and Doll{\'a}r, Piotr},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={10428--10436},
  year={2020}
}
```
