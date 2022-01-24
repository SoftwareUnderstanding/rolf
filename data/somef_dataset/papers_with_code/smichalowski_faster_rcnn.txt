# *Faster* R-CNN: Implementation of Matlab version in Python

### Introduction

**Faster** R-CNN is an object detection framework based on deep convolutional networks, which includes a Region Proposal Network (RPN) and an Object Detection Network. Both networks are trained for sharing convolutional layers for fast testing.

**This repo contains a Python implementation of Faster-RCNN originally developed in Matlab.
This code works with models trained using Matlab version of Faster-RCNN which is main difference between this and py-faster-rcnn.**

This code was developed for internal use in one of my projects at the end of 2015. I decided to publish it as is.

### Additional links

Faster R-CNN was initially described in an [arXiv tech report](http://arxiv.org/abs/1506.01497).

Faster R-CNN Matlab version is available at [faster-rcnn](https://github.com/ShaoqingRen/faster_rcnn).

Python version is available at [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn).


### Quickstart

Use provided Dockerfile to build container with all required dependencies.

1. Build docker container:
```bash
docker build -t faster_rcnn .
```

2. Download MATLAB Faster-RCNN models:
```bash
docker run --mount type=bind,source="$(pwd)",target=/app -w /app/models -it faster_rcnn /app/models/download_models.sh
```

3. Run detection
```bash
docker run --mount type=bind,source="$(pwd)",target=/app -w /app -it faster_rcnn python experiments/faster_rcnn.py models/000456.jpg faster_rcnn_VOC0712_ZF
```


### Resources

**Note**: This documentation may contain links to third party websites, which are provided for your convenience only. Such third party websites are not under Microsoft’s control. Microsoft does not endorse or make any representation, guarantee or assurance regarding any third party website, content, service or product. Third party websites may be subject to the third party’s terms, conditions, and privacy statements.

If the automatic "fetch_data" fails, you may manually download resouces from:

0. Final RPN+FastRCNN models: [OneDrive](https://onedrive.live.com/download?resid=D7AF52BADBA8A4BC!114&authkey=!AERHoxZ-iAx_j34&ithint=file%2czip), [DropBox](https://www.dropbox.com/s/jswrnkaln47clg2/faster_rcnn_final_model.zip?dl=0), [BaiduYun](http://pan.baidu.com/s/1hsFKmeK)


### License

Faster R-CNN is released under the MIT License (refer to the LICENSE file for details).
