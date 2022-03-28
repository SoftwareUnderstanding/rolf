# EfficientDet

[EfficientDet](https://arxiv.org/pdf/1911.09070.pdf) implementation for Object Detection using Tensorflow2

#### Generate Anchor (Optional)
* Generate optimal anchor for your dataset using [Anchor Optimizer](https://github.com/martinzlocha/anchor-optimization)
* Change the values of `ratios` and `scales` attribute of `AnchorParameters` class in `utils/util.py`

#### Train
* `cd` project directory
* Run `python setup.py build_ext --inplace`
* Change `classes` variable in `utils\config.py` based on your dataset
* Run `python train.py` for training using `imagenet` pretrained weights
* After training is finished, change `weight_path` variable value to trained weight path, example `weight_path=weights/D4/model1_0.1.h5`
* Run `python train.py` for final training

#### Test
* Run `python test.py`

#### Dataset structure
    ├── Dataset folder 
        ├── IMAGES
            ├── 1111.jpg
            ├── 2222.jpg
        ├── LABELS
            ├── 1111.xml
            ├── 2222.xml
        ├── train.txt
        ├── val.txt
#### Note 
* xml file should be in PascalVOC format
* for making `train.txt` and `val.txt`, see `VOC2012/ImageSets/Main/train.txt` 

#### Reference
* https://github.com/qubvel/efficientnet
* https://github.com/fizyr/keras-retinanet
* https://github.com/xuannianz/EfficientDet
* https://github.com/martinzlocha/anchor-optimization
* https://github.com/google/automl/tree/master/efficientdet
* https://github.com/tensorflow/models/tree/master/research/object_detection
