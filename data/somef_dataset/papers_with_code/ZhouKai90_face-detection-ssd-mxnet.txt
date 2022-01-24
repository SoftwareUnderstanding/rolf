# face detection SSD: Single Shot MultiBox Object Detector for face detection

SSD is an unified framework for object detection with a single network.

we use SSD to face detect and locate base on widerface dataset.

### Disclaimer
This is a modification on original SSD which is based on mxnet. The official
repository is available [here](https://github.com/apache/incubator-mxnet.git).
The arXiv paper is available [here](http://arxiv.org/abs/1512.02325).

### Demo results

![](https://github.com/ZhouKai90/face-detection-ssd-mxnet/blob/master/test/images/image%20(6)_detection.jpg)
![](https://github.com/ZhouKai90/face-detection-ssd-mxnet/blob/master/test/images/image%20(7)_detection.jpg)
![](https://github.com/ZhouKai90/face-detection-ssd-mxnet/blob/master/test/images/image%20(5)_detection.jpg)

### evaluation
Evaluate model base on FDDB
![](https://github.com/ZhouKai90/face-detection-ssd-mxnet/blob/master/evaluate/MAP.jpg)

### Try the demo

```
# cd /path/to/face-detection-ssd-mxnet
sh scripts/test.sh
```
### Train the model
This example only covers training on Pascal VOC dataset. Other datasets should
be easily supported by adding subclass derived from class `Imdb` in `dataset/imdb.py`.
See example of `dataset/pascal_voc.py` for details.
* Download the converted pretrained `vgg16_reduced` model [here](https://github.com/zhreshold/mxnet-ssd/releases/download/v0.2-beta/vgg16_reduced.zip), unzip `.param` and `.json` files
into `model/` directory by default.
* Download the widerface dataset into `data/`,
* modify `tools/widerfce2VOC.py`to your own parameters, and run to get the annotations.
* modify `tools/prepare_widerface_pascal.sh`to your own parameters, run to get the .rec  files for train.
* run `srcips/train.sh`
