# ClaRAN v0.2
[ClaRAN - Classifying Radio Galaxies Automatically with Neural Networks](https://academic.oup.com/mnras/article/482/1/1211/5142869)

# Faster R-CNN / Mask R-CNN on Radio Galaxy Zoo
As an upgrade of the [ClaRAN v0.1](https://github.com/chenwuperth/rgz_rcnn), ClaRAN v0.2 is based on the awesome [Tensorpack](https://github.com/tensorpack/tensorpack) project, which is likely the best-performing open source TensorFlow reimplementation of [Faster R-CNN](https://arxiv.org/abs/1506.01497). Moreover, [Tensorpack](https://github.com/tensorpack/tensorpack) has integrated the [Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144) (FPN). It also supports instance segmentation based on [Mask R-CNN](https://arxiv.org/abs/1703.06870), although ClaRAN v0.2 currently does not support source segmentation.

In summary, ClaRAN v0.2 includes the following new features:
+ The default "backbone" network is now ResNet50.
+ By default, FPN is used for multi-scale feature extraction
+ We further extended the image augmentation pipeline to support rotations (of both images and bounding boxes of sources) "on the fly"

With the above changes, ClaRAN has achieved an mAP of **86.1%** for D3 (83% for v0.1) and **85.9%** for D1 (79% for v0.1). The D3 dataset overlays radio contours onto Infrared maps, whereas the D1 dataset contains radio maps only. 

## Dependencies
+ Python 3.5+; OpenCV
+ TensorFlow ≥ 1.5 but < 2.0
+ TensorPack: `pip install --upgrade git+https://github.com/tensorpack/tensorpack.git`
+ pycocotools: `pip install pycocotools`
+ Pre-trained [ImageNet ResNet50 model](http://models.tensorpack.com/FasterRCNN/ImageNet-R50-AlignPadding.npz)
  from tensorpack model zoo
+ (Optional) Pre-trained [ClaRAN v0.2 RGZ D3 model](https://drive.google.com/open?id=1YRLu1fqdzuFR4SgdcA0dOXe_fPU1eWaD)
+ [RGZ data](https://drive.google.com/open?id=1x8ZkmuQrDdQdG_UVZPrWr0lj2dfxil3F), which needs to be organised as the following directory structure:
```
RGZ/DATA_DIR/
  annotations/
    instances_trainD3.json
    instances_testD3.json
  trainD3_hg/
    FIRSTJ23*_infraredct.png
  testD3_hg/
    FIRSTJ23*_infraredct.jpg
```

## Usage
### Train:
This is tested on both in-house GPU nodes and the Google Colab environment
```
python train.py --logdir ../train_logs/ --config \
        MODE_MASK=False MODE_FPN=True \
        DATA.BASEDIR=./data \
        BACKBONE.WEIGHTS=./weights/pretrained/ImageNet-R50-AlignPadding.npz \
        DATA.TRAIN=trainD3_hg DATA.VAL=testD3_hg \
        PREPROC.TRAIN_SHORT_EDGE_SIZE=600,600 \
        PREPROC.TEST_SHORT_EDGE_SIZE=600 \
	      TRAIN.LR_SCHEDULE=20000,30000,40000
```

### Inference:

To detect all radio sources on a D3 image (needs DISPLAY to show the outputs):
```
python train.py --predict ./data/testD3_hg/FIRSTJ235752.4+101110_infraredct.png \
                --load ./weights/rgz_models/d3/model-70000.data-00000-of-00001 \
                --config MODE_MASK=False MODE_FPN=True \
                        DATA.BASEDIR=./data \
        BACKBONE.WEIGHTS=../weights/pretrained/ImageNet-R50-AlignPadding.npz \
        DATA.TRAIN=trainD3_hg DATA.VAL=testD3_hg \
        PREPROC.TRAIN_SHORT_EDGE_SIZE=600,600 \
        PREPROC.TEST_SHORT_EDGE_SIZE=600 \
	      TEST.RESULT_SCORE_THRESH_VIS=0.7 \
	      TEST.RESULT_SCORE_THRESH=0.7 \
```
If the DISPLAY is not set, it will produce an PNG image under the current directory:

<img src="output_claran.png">

On the left is the original D3 image, and the detected sources are shown on the right.

Notice that the ``--load`` argument loads the pre-trained D3 model (470 MB) that can also be [downloaded](https://drive.google.com/open?id=1YRLu1fqdzuFR4SgdcA0dOXe_fPU1eWaD) if you want to skip the training (A couple of hours using four P100 GPUs) altogether.

## Results

Some result comparisons between different methods and architectures will be posted here hopefully soon.
