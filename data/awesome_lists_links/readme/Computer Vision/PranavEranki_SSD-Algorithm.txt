# Object Detection using Single Shot MultiBox Detector
*Author : Pranav Eranki 2018*

Creating a neural network for object detection that has high speed and accuracy --> utilized the Single Shot MultiBox Detector (SSD) algorithm.

The original paper about the Single Shot MultiBox Detector can be found at https://arxiv.org/pdf/1512.02325.pdf.

This was a project made for fun and for exploring further the creation and usage of the SSD.

__Extension of AI Workshop 9__

## Project Structure

```
.
├─ data/
│  ├─ raw/                   <- downloaded archives
│  │  ├─ voc2007/
│  │  └─ voc2012/
│  ├─ interim/               <- extracted archives
│  └─ processed/             <- converted datasets
├─ models/                   <- pre-trained weights and frozen models
│  └─ vgg_16_imagenet/       <- pre-trained VGG 16 weights
├─ src/
│  ├─ data/                  <- data input pipeline
│  │  └─ preprocessors/      <- data pre-processors
│  ├─ datasets/              <- extract and convert datasets
│  │  └─ common/
│  ├─ models/                <- model implementation
│  │  ├─ custom_layers/
│  │  └─ ssd/
│  ├─ utils/                 <- utility functions and classes
│  │  └─ common/
│  ├─ eval.py                <- evaluate a model using mean average precision
│  ├─ freeze.py              <- freeze a trained model for faster inference
│  ├─ infer_live_opencv.py   <- start inference with video or webcam using OpenCV
│  ├─ infer_live.py          <- start inference with video or webcam using Matplotlib
│  ├─ infer.py               <- start inference with a single image
│  └─ train.py               <- train a new model
├─ training/                 <- run configurations and saved checkpoints
│  └─ run_*/                    created by src/train.py
├─ LICENSE.md
└─ README.md
```

## Getting Started

To get started, download the pre-trained [VGG 16 weights](http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz). Move the file `vgg_16.ckpt` to `tf-ssd-vgg/models/vgg_16_imagenet`.

To train the neural network, download the below and move them to `tf-ssd-vgg/data/raw/voc2007` and `tf-ssd-vgg/data/raw/voc2012`.
- [Pascal VOC 2007 trainval](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar)
- [Pascal VOC 2007 test](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar)
- [Pascal VOC 2012 trainval](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)

## Dependencies

The project was compiled using the following packages:
- **Matplotlib** 2.2.2 ([Information](https://matplotlib.org/))
- **NumPy** 1.14.5 ([Information](https://www.numpy.org/))
- **OpenCV** 3.4.1 ([Information](https://opencv.org/))
- **Tensorflow GPU** 1.9.0 ([Information](https://www.tensorflow.org/))
- **tqdm** 4.23.4 ([Information](https://github.com/tqdm/tqdm))

## LICENSE

All python code in this repository is available under an MIT license.
