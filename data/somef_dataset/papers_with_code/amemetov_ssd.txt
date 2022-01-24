# **Single Shot MultiBox Detector**

---

[//]: # (Image References)
[loss-curve]: ./loss-curve.png "Loss Curve"

**Keras/Tensorflow implementation of the [SSD](https://arxiv.org/abs/1512.02325)**

The base source code is placed in the [ssd dir](/ssd).
Currently **SSD300** NN based on **VGG16** model is implemented - see [ssd300_vgg16.py](ssd/ssd300_vgg16.py).

## **Dataset**
The **SSD300-VGG16** is trained on [Pascal VOC 2007+2012 dataset](http://host.robots.ox.ac.uk/pascal/VOC/).
Initially Ground-Truth Boxes are fetched from Pascal VOC dataset and stored as a hashtable.
The keys of the hashtable are filenames, 
values are numpy arrays containing normalized bounding boxes, one-hot-encoded classes and difficulty property:
[xmin, ymin, xmax, ymax, one-hot-encoded-class, is-difficult].
Ground-Truth Boxes are stored in the following pickle files:
* [pascal_voc_2007_test.p](data/pascal_voc_2007_test.p) (see http://host.robots.ox.ac.uk/pascal/VOC/voc2007/#testdata)
* [pascal_voc_2007_trainval.p](data/pascal_voc_2007_trainval.p) (see http://host.robots.ox.ac.uk/pascal/VOC/voc2007/#devkit)
* [pascal_voc_2012_trainval.p](data/pascal_voc_2012_trainval.p) (see http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit)

## **PriorBoxes (a.k.a. DefaultBoxes and Anchors)**
PriorBoxes are generated as in [the origin Caffe implementation](https://github.com/weiliu89/caffe/blob/ssd/examples/ssd/ssd_pascal.py).

[PriorBoxes.ipynb](tests/PriorBoxes.ipynb) contains samples of how PriorBoxes might look.

## **Data Augmentation**
See [DataAugmentation.ipynb](tests/DataAugmentation.ipynb) for the whole process samples.
See [Imaging.ipynb](tests/Imaging.ipynb) for photo-metric distortion samples.

## **Hard negative mining**
Mining hard examples is implemented in [the SsdLoss class](ssd/losses.py)

## **Training**
[SSD300-VGG16.ipynb](SSD300-VGG16.ipynb) contains the training process.

## **Evaluating**
This implementation has a lower performance comparing to the original Caffe implementation:
**overall mAP = 66%**

| Class  | AP (%) |
|:-------------:|:-------------:| 
| aeroplane |   76 |
| bicycle   |   76 |
| bird      |   66 |
| boat      |   63 |
| bottle    |   39 |
| bus       |   71 |
| car       |   80 |
| cat       |   80 |
| chair     |   36 |
| cow       |   59 |
| diningtable | 64 |
| dog         | 70 |
| horse       | 75 |
| motorbike   | 72 |
| person      | 70 |
| pottedplant | 44 |
| sheep       | 56 |
| sofa        | 71 |
| train       | 76 |
| tvmonitor   | 68 |

## **Dependencies**
See [Anaconda env file](./env/anaconda-dl-env.yml) for dependencies.

To create env use:
```bash
conda env create -f anaconda-dl-env.yml
```

## **References**

1. [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325)

2. [The origin SSD Caffe implementation](https://github.com/weiliu89/caffe/tree/ssd)

3. [mAP calculating in py-faster-rcnn project](https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py)







