# R<sup>3</sup>Det: Refined Single-Stage Detector with Feature Refinement for Rotating Object

## Abstract
- [R<sup>3</sup>Det](https://arxiv.org/abs/1908.05612) is based on [Focal Loss for Dense Object Detection](https://arxiv.org/pdf/1708.02002.pdf), and it is completed by [YangXue](https://yangxue0827.github.io/) and [ZhangGeFan](https://github.com/zhanggefan).
- MMDetection is an open source object detection toolbox based on PyTorch. It is a part of the OpenMMLab project developed by [Multimedia Laboratory, CUHK](http://mmlab.ie.cuhk.edu.hk/).

Techniques:     
- [x] [ResNet](https://arxiv.org/abs/1512.03385)
- [x] CUDA version of [Feature Refinement Module (FRM)](https://arxiv.org/abs/1908.05612) as PyTorch extension

## Pipeline
![5](pipeline.png)

## Performance
### DOTA1.0
| Model |    Backbone    |    Training data    |    Val data    |    mAP  | Model Link  | GPU | Image/GPU | Anchor | Reg. Loss| lr schd | Data Augmentation | Configs |       
|:------------:|:------------:|:------------:|:---------:|:-----------:|:----------:|:----------:|:-----------:|:---------:|:---------:|:---------:|:---------:|:---------:|     
| [R<sup>3</sup>Det*](https://arxiv.org/abs/1908.05612)| ResNet50 600->800 | DOTA1.0 trainval | DOTA1.0 test | 71.90 | [Google Drive](https://drive.google.com/file/d/1JYyEHyzloRcxYlSRuiCLEaCjjj4Q7L_E/view?usp=sharing) -- [Baidu Drive (u8bj)](https://pan.baidu.com/s/1Ijmh1Lco4T7HPwAtT2h0Zg) | **1X** GeForce RTX 2080 Ti | 6 | H + R | smooth L1 | 2x | No | [r3det_r50_fpn_2x_CustomizeImageSplit.py](./configs/r3det/r3det_r50_fpn_2x_CustomizeImageSplit.py) |

[R<sup>3</sup>Det*](https://arxiv.org/abs/1908.05612): R<sup>3</sup>Det with two refinement stages
                 
## Compile
```
python setup.py install
```

## Train
```
sh rtools/train.sh
```
Or equivalent command:   
```
python tools/train.py {configuration-file-path}
```
Before training, please:
1. Change the paths in lines 97-98 & 102-103 of [dota_image_split.py](./rtools/dota_image_split.py) according to your local DOTA dataset directory.
2. Run `python dota_image_split.py` to crop train & val set images into smaller tiles, and generate per-tile label files into the directories you specified in step [1].
3. Change the lines 4-10 of [dotav1_rotational_detection.py](./configs/r3det/datasets/dotav1_rotational_detection.py). Paths in lines 5-8 shall direct to the folders containing the cropped image tiles and label files generated in step [2].
4. Have fun with `sh rtools/train.sh` and watch the model train!
   
## Test
```
sh rtools/test.sh
```
Or equivalent command:   
```
python tools/test.py {configuration-file-path} {checkpoint-file-path} --format-only --options submission_dir={path-to-save-submission-files}
```
Before test, please make sure the checkpoint file path (in ```rtools/test.sh```) is correct.

## Citation

If this is useful for your research, please consider cite.

```
@article{yang2019r3det,
    title={R3Det: Refined Single-Stage Detector with Feature Refinement for Rotating Object},
    author={Yang, Xue et al},
    journal={arXiv preprint arXiv:1908.05612},
    year={2019}
}

@inproceedings{xia2018dota,
    title={DOTA: A large-scale dataset for object detection in aerial images},
    author={Xia, Gui-Song and Bai, Xiang and Ding, Jian and Zhu, Zhen and Belongie, Serge and Luo, Jiebo and Datcu, Mihai and Pelillo, Marcello and Zhang, Liangpei},
    booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    pages={3974--3983},
    year={2018}
}

@article{mmdetection,
  title   = {{MMDetection}: Open MMLab Detection Toolbox and Benchmark},
  author  = {Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and
             Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and
             Liu, Ziwei and Xu, Jiarui and Zhang, Zheng and Cheng, Dazhi and
             Zhu, Chenchen and Cheng, Tianheng and Zhao, Qijie and Li, Buyu and
             Lu, Xin and Zhu, Rui and Wu, Yue and Dai, Jifeng and Wang, Jingdong
             and Shi, Jianping and Ouyang, Wanli and Loy, Chen Change and Lin, Dahua},
  journal= {arXiv preprint arXiv:1906.07155},
  year={2019}
}
```

## Reference
- https://github.com/Thinklab-SJTU/R3Det_Tensorflow    
- https://github.com/open-mmlab/mmdetection     
- https://github.com/endernewton/tf-faster-rcnn   
- https://github.com/zengarden/light_head_rcnn   
- https://github.com/tensorflow/models/tree/master/research/object_detection    
- https://github.com/fizyr/keras-retinanet  

