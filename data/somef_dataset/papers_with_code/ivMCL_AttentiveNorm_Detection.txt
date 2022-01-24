# Attentive Normalization

**Please check our refactorized code at [iVMCL-Release](https://github.com/iVMCL/iVMCL-Release).**

This repo contains the code and pretrained models for [AOGNets: Compositional Grammatical Architectures for Deep Learning
](https://arxiv.org/abs/1711.05847)(CVPR 2019) and [Attentive Normalization](https://arxiv.org/abs/1908.01259). The models are trained on COCO object detection and instance segmentation task with Mask-RCNN and Cascade-Mask-RCNN model. We replace the backbone with [our imagenet pretrained backbones](https://github.com/iVMCL/AOGNet-v2) and head normalization with our Attentive Normalization. The results and trained models could be found in the table below. 

```
@article{li2019attentive,
  title={Attentive Normalization},
  author={Li, Xilai and Sun, Wei and Wu, Tianfu},
  journal={arXiv preprint arXiv:1908.01259},
  year={2019}
}

@inproceedings{li2019aognets,
  title={AOGNets: Compositional Grammatical Architectures for Deep Learning},
  author={Li, Xilai and Song, Xi and Wu, Tianfu},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={6220--6230},
  year={2019}
}
```
## Getting Started
- The ImageNet pretrained models are needed for training, and they can be downloaded from the [https://github.com/iVMCL/AOGNet-v2](https://github.com/iVMCL/AOGNet-v2). 

```
git clone https://github.com/iVMCL/AttentiveNorm_Detection.git
cd AttentiveNorm_Detection
mkdir pretrained_models   # And put all the pretrained models under that directory.
```

- Installation, data preparation, and training/evaluating models are the same as it in the [original mmdetection](https://github.com/open-mmlab/mmdetection) repo. 

## Results and Models
Mask-RCNN
<table>
  <tr>
    <th>Architecture</th>
    <th>Backbone</th>
    <th>Head</th>
    <th>#Params</th>
    <th>box AP</th>
    <th>mask AP</th>
    <th>Download</th>
  </tr>
  <tr>
    <td rowspan="5">ResNet-50</td>
    <td>BN</td>
    <td>-</td>
    <td>45.71M</td>
    <td>39.2</td>
    <td>35.2</td>
    <td><a href="https://drive.google.com/open?id=1m6_-5ri-ZcVtcrazAIXaPr_0y8KSx42p">Google Drive</a></td>
  </tr>
  <tr>
    <td>AN (w/ BN)</td>
    <td>-</td>
    <td>45.91M</td>
    <td>40.8</td>
    <td>36.4</td>
    <td><a href="https://drive.google.com/open?id=12bAq3OhZ-jnv_jGNsVb4TsFhzzueDCDh">Google Drive</a></td>
  </tr>
    <tr>
    <td>*GN</td>
    <td>GN</td>
    <td>45.72M</td>
    <td>40.3</td>
    <td>35.7</td>
    <td>-</td>
  </tr>
    <tr>
    <td>*SN</td>
    <td>SN</td>
    <td>-</td>
    <td>41.0</td>
    <td>36.5</td>
    <td>-</td>
  </tr>
    <tr>
    <td>AN (w/ BN)</td>
    <td>AN (w/ GN)</td>
    <td>45.96M</td>
    <td>41.6</td>
    <td>37.4</td>
    <td><a href="https://drive.google.com/open?id=1pFrwDMERNAnvHZw73fdHqa400-ou-BW9">Google Drive</a></td>
  </tr>
  <tr>
    <td rowspan="4">ResNet-101</td>
    <td>BN</td>
    <td>-</td>
    <td>64.70M</td>
    <td>41.4</td>
    <td>36.8</td>
    <td><a href="https://drive.google.com/open?id=1xWEsfHiwZp_INCvygJMNy6amESVmXqRn">Google Drive</a></td>
  </tr>
  <tr>
    <td>AN (w/ BN)</td>
    <td>-</td>
    <td>65.15M</td>
    <td>43.1</td>
    <td>38.2</td>
    <td><a href="https://drive.google.com/open?id=1A9Dp_DMbMu7j4D6Xl9Ej7Q6_SbkdCs3f">Google Drive</a></td>
  </tr>
    <tr>
    <td>*GN</td>
    <td>GN</td>
    <td>64.71M</td>
    <td>41.8</td>
    <td>36.8</td>
    <td>-</td>
  </tr>
  <tr>
    <td>AN (w/ BN)</td>
    <td>AN (w/ GN)</td>
    <td>65.20M</td>
    <td>43.2</td>
    <td>38.8</td>
    <td><a href="https://drive.google.com/open?id=1ySGHIuYDlcPxlXlsNzwqgF_kDjDbetSK">Google Drive</a></td>
  </tr>
  <tr>
    <td rowspan="3">AOGNet-12m</td>
    <td>BN</td>
    <td>-</td>
    <td>33.09M</td>
    <td>40.7</td>
    <td>36.4</td>
    <td><a href="https://drive.google.com/open?id=1zwFvFH2h6jNgrwSW5x47zj1k1VAEcJUt">Google Drive</a></td>
  </tr>
  <tr>
    <td>AN (w/ BN)</td>
    <td>-</td>
    <td>33.21M</td>
    <td>42.0</td>
    <td>37.8</td>
    <td><a href="https://drive.google.com/open?id=1DfDd0D81TV3JpKbGkX_X5stk1bVdF5GR">Google Drive</a></td>
  </tr>
  <tr>
    <td>AN (w/ BN)</td>
    <td>AN (w/ GN)</td>
    <td>33.26M</td>
    <td>43.0</td>
    <td>38.7</td>
    <td><a href="https://drive.google.com/open?id=1s_WO4O9hBjYMLHRffcCo8Qb0xt1_3ifR">Google Drive</a></td>
  </tr>
  <tr>
    <td rowspan="3">AOGNet-40m</td>
    <td>BN</td>
    <td>-</td>
    <td>60.73M</td>
    <td>43.4</td>
    <td>38.5</td>
    <td><a href="https://drive.google.com/open?id=1LIZDL22XcaUlPAKdQkPa8INB9zTTtrNj">Google Drive</a></td>
  </tr>
  <tr>
    <td>AN (w/ BN)</td>
    <td>-</td>
    <td>60.97M</td>
    <td>44.1</td>
    <td>39.0</td>
    <td><a href="https://drive.google.com/open?id=1XvRgjqCxBE5iiDcm-s10piMPuaglQrSa">Google Drive</a></td>
  </tr>
  <tr>
    <td>AN (w/ BN)</td>
    <td>AN (w/ GN)</td>
    <td>61.02M</td>
    <td>44.9</td>
    <td>40.2</td>
    <td><a href="https://drive.google.com/open?id=1QbVVHi_bE_tEIR0H8u-1FXa2jCzN7GvA">Google Drive</a></td>
  </tr>
</table>


Cascade Mask-RCNN

<table>
  <tr>
    <th>Architecture</th>
    <th>Backbone</th>
    <th>Head</th>
    <th>#Params</th>
    <th>box AP</th>
    <th>mask AP</th>
    <th>Download</th>
  </tr>
  <tr>
    <td rowspan="2">ResNet-101</td>
    <td>BN</td>
    <td>-</td>
    <td>96.32M</td>
    <td>44.4</td>
    <td>38.2</td>
    <td><a href="https://drive.google.com/open?id=1OxMfuXEmJJTMS2LgA8at54KqYo3lrPGr">Google Drive</a></td>
  </tr>
  <tr>
    <td>AN (w/ BN)</td>
    <td>-</td>
    <td>96.77M</td>
    <td>45.8</td>
    <td>39.6</td>
    <td><a href="https://drive.google.com/open?id=1J69Rg9Dl5tl3xoVljSMYZzzpouHDJ3Ty">Google Drive</a></td>
  </tr>
  <tr>
    <td rowspan="2">AOGNet-40m</td>
    <td>BN</td>
    <td>-</td>
    <td>92.35M</td>
    <td>45.6</td>
    <td>39.3</td>
    <td><a href="https://drive.google.com/open?id=19pyLHeFR_DQuWIS15VopV8wF6KJv9dQM">Google Drive</a></td>
  </tr>
  <tr>
    <td>AN (w/ BN)</td>
    <td>-</td>
    <td>92.58M</td>
    <td>46.5</td>
    <td>40.0</td>
    <td><a href="https://drive.google.com/open?id=1BC99jyKxeX5vB1fBqhIPFvBbP4YlEk9f">Google Drive</a></td>
  </tr>
</table>

# MMDetection

**News**: We released the technical report on [ArXiv](https://arxiv.org/abs/1906.07155).

## Introduction

The master branch works with **PyTorch 1.1** or higher.

mmdetection is an open source object detection toolbox based on PyTorch. It is
a part of the open-mmlab project developed by [Multimedia Laboratory, CUHK](http://mmlab.ie.cuhk.edu.hk/).

![demo image](demo/coco_test_12510.jpg)

### Major features

- **Modular Design**

  We decompose the detection framework into different components and one can easily construct a customized object detection framework by combining different modules.

- **Support of multiple frameworks out of box**

  The toolbox directly supports popular and contemporary detection frameworks, *e.g.* Faster RCNN, Mask RCNN, RetinaNet, etc.

- **High efficiency**

  All basic bbox and mask operations run on GPUs now. The training speed is faster than or comparable to other codebases, including [Detectron](https://github.com/facebookresearch/Detectron), [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) and [SimpleDet](https://github.com/TuSimple/simpledet).

- **State of the art**

  The toolbox stems from the codebase developed by the *MMDet* team, who won [COCO Detection Challenge](http://cocodataset.org/#detection-leaderboard) in 2018, and we keep pushing it forward.

Apart from MMDetection, we also released a library [mmcv](https://github.com/open-mmlab/mmcv) for computer vision research, which is heavily depended on by this toolbox.

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Updates

v1.0rc0 (27/07/2019)
- Implement lots of new methods and components (Mixed Precision Training, HTC, Libra R-CNN, Guided Anchoring, Empirical Attention, Mask Scoring R-CNN, Grid R-CNN (Plus), GHM, GCNet, FCOS, HRNet, Weight Standardization, etc.). Thank all collaborators!
- Support two additional datasets: WIDER FACE and Cityscapes.
- Refactoring for loss APIs and make it more flexible to adopt different losses and related hyper-parameters.
- Speed up multi-gpu testing.
- Integrate all compiling and installing in a single script.

v0.6.0 (14/04/2019)
- Up to 30% speedup compared to the model zoo.
- Support both PyTorch stable and nightly version.
- Replace NMS and SigmoidFocalLoss with Pytorch CUDA extensions.

v0.6rc0(06/02/2019)
- Migrate to PyTorch 1.0.

v0.5.7 (06/02/2019)
- Add support for Deformable ConvNet v2. (Many thanks to the authors and [@chengdazhi](https://github.com/chengdazhi))
- This is the last release based on PyTorch 0.4.1.

v0.5.6 (17/01/2019)
- Add support for Group Normalization.
- Unify RPNHead and single stage heads (RetinaHead, SSDHead) with AnchorHead.

v0.5.5 (22/12/2018)
- Add SSD for COCO and PASCAL VOC.
- Add ResNeXt backbones and detection models.
- Refactoring for Samplers/Assigners and add OHEM.
- Add VOC dataset and evaluation scripts.

v0.5.4 (27/11/2018)
- Add SingleStageDetector and RetinaNet.

v0.5.3 (26/11/2018)
- Add Cascade R-CNN and Cascade Mask R-CNN.
- Add support for Soft-NMS in config files.

v0.5.2 (21/10/2018)
- Add support for custom datasets.
- Add a script to convert PASCAL VOC annotations to the expected format.

v0.5.1 (20/10/2018)
- Add BBoxAssigner and BBoxSampler, the `train_cfg` field in config files are restructured.
- `ConvFCRoIHead` / `SharedFCRoIHead` are renamed to `ConvFCBBoxHead` / `SharedFCBBoxHead` for consistency.

## Benchmark and model zoo

Supported methods and backbones are shown in the below table.
Results and models are available in the [Model zoo](docs/MODEL_ZOO.md).

|                    | ResNet   | ResNeXt  | SENet    | VGG      | HRNet |
|--------------------|:--------:|:--------:|:--------:|:--------:|:-----:|
| RPN                | ✓        | ✓        | ☐        | ✗        | ✓     |
| Fast R-CNN         | ✓        | ✓        | ☐        | ✗        | ✓     |
| Faster R-CNN       | ✓        | ✓        | ☐        | ✗        | ✓     |
| Mask R-CNN         | ✓        | ✓        | ☐        | ✗        | ✓     |
| Cascade R-CNN      | ✓        | ✓        | ☐        | ✗        | ✓     |
| Cascade Mask R-CNN | ✓        | ✓        | ☐        | ✗        | ✓     |
| SSD                | ✗        | ✗        | ✗        | ✓        | ✗     |
| RetinaNet          | ✓        | ✓        | ☐        | ✗        | ✓     |
| GHM                | ✓        | ✓        | ☐        | ✗        | ✓     |
| Mask Scoring R-CNN | ✓        | ✓        | ☐        | ✗        | ✓     |
| FCOS               | ✓        | ✓        | ☐        | ✗        | ✓     |
| Double-Head R-CNN  | ✓        | ✓        | ☐        | ✗        | ✓     |
| Grid R-CNN (Plus)  | ✓        | ✓        | ☐        | ✗        | ✓     |
| Hybrid Task Cascade| ✓        | ✓        | ☐        | ✗        | ✓     |
| Libra R-CNN        | ✓        | ✓        | ☐        | ✗        | ✓     |
| Guided Anchoring   | ✓        | ✓        | ☐        | ✗        | ✓     |

Other features
- [x] DCNv2
- [x] Group Normalization
- [x] Weight Standardization
- [x] OHEM
- [x] Soft-NMS
- [x] Generalized Attention
- [x] GCNet
- [x] Mixed Precision (FP16) Training


## Installation

Please refer to [INSTALL.md](docs/INSTALL.md) for installation and dataset preparation.


## Get Started

Please see [GETTING_STARTED.md](docs/GETTING_STARTED.md) for the basic usage of MMDetection.

## Contributing

We appreciate all contributions to improve MMDetection. Please refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md) for the contributing guideline.

## Acknowledgement

MMDetection is an open source project that is contributed by researchers and engineers from various colleges and companies. We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks.
We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new detectors.


## Citation

If you use this toolbox or benchmark in your research, please cite this project.

```
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


## Contact

This repo is currently maintained by Kai Chen ([@hellock](http://github.com/hellock)), Jiangmiao Pang ([@OceanPang](https://github.com/OceanPang)), Jiaqi Wang ([@myownskyW7](https://github.com/myownskyW7)) and Yuhang Cao ([@yhcao6](https://github.com/yhcao6)).
