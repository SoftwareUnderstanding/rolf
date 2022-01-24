# Airbus Ships detection problem

[![NPM Version][npm-image]][npm-url]
[![Build Status][travis-image]][travis-url]
[![Downloads Stats][npm-downloads]][npm-url]

This a computer vision object detection and segmentation problem on kaggle (https://www.kaggle.com/c/airbus-ship-detection#description). In this problem, I build a model that detects all ships in satellite images and generate a mask for each ship. There several deep learning models that works with image detection such as YOLO, R-CNN, Fast R-CNN, Faster R-CNN. For objection segmentation, Unet is a great tools. Recently there is a nice paper on object instance segmentation (https://arxiv.org/abs/1703.06870) called Mask R-CNN.

In this problem, most image (~80%) contains no ships. So my strategy is the following:

1. I build a classifier to detect if a image has any ships.
2. Feed the image that contains image detected by the classifier to Mask R-CNN.

## Results

<p align="center">
  <img src="seg_val0.png">
</p>

### Acknowledgement
This code is implemented on maskrcnn frameworks (https://github.com/matterport/Mask_RCNN). Thanks for their great work!

### Prerequisites

Python 3.6

Jupyter Notebook

### Meta

[Chi Zhang](https://zhangchi9.github.io/) – [@LinkedIn](https://www.linkedin.com/in/chi-zhang-2018/) – c.zhang@neu.edu

Distributed under the MIT license. See [LICENSE](https://github.com/zhangchi9/Ames-Iowa-house-prices-prediction/blob/master/LICENSE) for more information.


<!-- Markdown link & img dfn's -->
[mit-url]:https://opensource.org/licenses/MIT
[mit-image]:https://img.shields.io/badge/License-MIT-yellow.svg
[npm-image]: https://img.shields.io/npm/v/datadog-metrics.svg?style=flat-square
[npm-url]: https://npmjs.org/package/datadog-metrics
[npm-downloads]: https://img.shields.io/npm/dm/datadog-metrics.svg?style=flat-square
[travis-image]: https://img.shields.io/travis/dbader/node-datadog-metrics/master.svg?style=flat-square
[travis-url]: https://travis-ci.org/dbader/node-datadog-metrics
[wiki]: https://github.com/yourname/yourproject/wiki

