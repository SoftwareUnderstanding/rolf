# A Fork of PyTorch Implemation of YOLOv3 to Accomodate Custom Data

**This fork is a work in progress.  It will be noted here when this is ready for broader, more production, use.  Issues are welcome.**

## General Updates/Improvements

* User may bring custom data with a custom number of classes
* Code cleaner and parameterized
* Training has fine-tuning
* For more details on updates and status of this fork, see notes at the bottom of the README.md

_We love you COCO, but we have our own interests now._

This project is a "You Only Look Once" v3 sample using PyTorch, a fork of https://github.com/ayooshkathuria/pytorch-yolo-v3, with updates and improvements specifically for architecture on custom data labeled with VoTT (versus the classic download of VOC or COCO data and pre-existing labels).  This fork allows the user to **bring their own dataset**.

<img src="imgs/id_plumeria_sml.png" width="70%" align="center">

IMPORTANT NOTES
--- 
* This project is a work in progress and issues are welcome (it's also a hobby at this point, so updates may be slow)
* There are two phases in training: 1) the first pass (set number of epochs in `cfg` file and layers to train on set on command line) and 2) fine-tuning (all parameters are trained upon, i.e. all layers "opened up")
* Training is very sensitive to the amount of layers to unfreeze for transfer learning which is set on command line (try more, then work down to less - affects first pass); if the loss does not decrease/model converge, try opening up more layers to be trained upon
* Training is sensitive to initial LR and LR decreases (the schedule)
* The example config files are 1 and 2 class, see below on how to change the number of classes
* Always calculate your own anchors (i.e. anchor box width and heights)

## Setup

* Install the required Python packages (`pip install -r requirements.txt`).
* Download the [full YOLO v3 (237 MB)](https://pjreddie.com/media/files/yolov3.weights) or [tiny YOLO v3 (33.8 MB)](https://pjreddie.com/media/files/yolov3-tiny.weights) model.  **Fun fact:  this project utilizes the weights originating in Darknet format**.

## Collect and Label Data

1. Use one of these two labeling tools and export to YOLO format:
  * <a href="https://github.com/microsoft/VoTT/releases/tag/v1.7.1" target="_blank">VoTT v1 download</a> and <a href="https://github.com/microsoft/VoTT/tree/ec6057c4c95780f7547d5c55245c6f48b396e29c" target="_blank">v1 README</a> - labeling tool to create bounding boxes around objects of interest in images and export to YOLO format.
  * <a href="https://github.com/tzutalin/labelImg" target="_blank">labelIMG</a> (probably easiest to `pip3 install labelImg`)
2. If you wish to train on all labeled images, make sure they are all in the `train.txt` file (this is read by the `customloader.py`).

The `data` output folder should be a subdirectory here with the images, labels and pointer file.

## Train Model

### Modifications for Custom

To summarize, within the appropriate config file located under the `cfg` folder (note, examples are there), the following properties will be modified as per instructions here.

in the `[net]` part at the beginning (`steps` are epochs, here), e.g.:

```
classes=1
batch=2
steps=1
anchors = 25,87, 44,55, 46,110, 72,74, 80,118, 93,45, 105,72, 127,96, 144,58
```

in `[yolo]` parts, e.g.:

```
anchors = 25,87, 44,55, 46,110, 72,74, 80,118, 93,45, 105,72, 127,96, 144,58
classes=1
```

in the `[convolutional]` parts above `[yolo]` layers, e.g.:

```
filters=18
```

Keep reading to find out more.

**Changing Anchors**

The tiny architecture has 6 anchors, whereas, the non-tiny or full sized YOLOv3 architecture has 9 anchors (or anchor boxes).  

* Run Kmeans algorithm:  these anchors should be manually discovered with and specified in the `cfg` file.  Run `scripts/convert_labels.py` and then `kmeans.py` on new annotatino format output.  (a workaround for now to get anchors)
* Modify the `anchors` in the `yolov3-tiny-x.cfg` or `yolov3-x.cfg` in the `[net]` section and the `[yolo]` sections with the new anchor box x, y values.

**Changing Filters and Classes**

* Ensure the `yolov3-tiny-x.cfg` or `yolov3-x.cfg` is set up correctly.  Note, the number of classes will affect the last convolutional layer filter numbers (conv layers before the yolo layer) as well as the yolo layers themselves - so **will need to be modified manually** to suit the needs of the user.

* Change the number of classes appropriately (e.g. `classes=2`) in each `[yolo]` layer (there will be three in the yolov3 and 2 in yolov3-tiny config files).

* Modify the filter number of the CNN layer directly before each [yolo] layer to be:  `filters=`, then calculate (classes + 5)x3, and place after.  So, if `classes=1` then should be `filters=18`. If `classes=2` then write `filters=21`, and so on.

**Additional Instructions**

* Create a list of the training images file paths, one per line, called `train.txt` and place it in the `data` folder.  e.g.

`train.txt`
```
imgs/482133.JPG
imgs/482128.JPG
imgs/482945.jpg
imgs/483153.JPG
imgs/481427.jpg
imgs/480836.jpg
imgs/483522.JPG
imgs/482535.JPG
imgs/483510.JPG
```

### Run Training Script

Cmd example:

    python train.py --cfg cfg/yolov3-2class.cfg --weights yolov3.weights --datacfg data/obj.data --lr 0.0005 --unfreeze 2

Usage:

    python train.py --help

## Inference

Here, you will use your trained model for evaluation on test data and a live video analysis.  The folder `runs` is where trained models get saved by default under a date folder.

**Additional Instructions**

* Create a list of the test images file paths, one per line, called `test.txt` and place it in the `data` folder.  e.g.

`test.txt`
```
imgs/482308.JPG
imgs/483367.JPG
imgs/483037.jpg
imgs/481962.JPG
imgs/481472.jpg
imgs/483303.JPG
imgs/483326.JPG
```

### Evaluation

Cmd example:

    python eval.py --cfg cfg/yolov3-2class.cfg --weights runs/<your trained model>.pth --overlap 0.3

Usage:

    python eval.py --help

### Run Video Detection

Cmd example:

    python live.py --cfg cfg/yolov3-tiny.cfg --weights runs/<your trained model>.pth --datacfg data/obj.data --confidence 0.6

Usage:
    
    python live.py --help

## Helpful Definitions

- YOLOv3:  You Only Look Once v3.  Improvments over v1, v2 and YOLO9000 which include [Ref](https://towardsdatascience.com/yolo-v3-object-detection-53fb7d3bfe6b):
  - Predicts more bounding boxes per image (hence a bit slower)
  - Detections at 3 scales
  - Addressed issue of detecting small objects
  - New loss function (cross-entropy replaces squared error terms)
  - Can perform multi-label classification (no more mutually exclusive labels)
  - Performance on par with other architectures (a bit faster than SSD, even)
- Tiny-YOLOv3:  A reduced network architecture for smaller models designed for mobile, IoT and edge device scenarios
- Anchors:  There are 5 anchors per box.  The anchor boxes are designed for a specific dataset using K-means clustering, i.e., a custom dataset must use K-means clustering to generate anchor boxes.  It does not assume the aspect ratios or shapes of the boxes. [Ref](https://medium.com/@vivek.yadav/part-1-generating-anchor-boxes-for-yolo-like-network-for-vehicle-detection-using-kitti-dataset-b2fe033e5807)
- Loss, `loss.backward()` and `nn.MSELoss` (for loss confidence):  Mean Squared Error
- IOU:  intersection over union between predicted bounding boxes and ground truth boxes

**The original YOLOv3 paper by Joseph Redmon and Ali Farhadi:  https://arxiv.org/pdf/1804.02767.pdf**

Updates to original codebase
---
* [x] Made it possible to bring any image data for object detection with `customloader.py` (using <a href="https://github.com/Microsoft/VoTT" target="_blank">VoTT</a> to label)
* [x] Replace CUDA flag in lieu of the simple `tensor_xyz.to(device)` method
* [x] Fix `customloader.py` to take multiple classes as a parameter in config file (e.g. `yolov3-2class.cfg`)
* [x] Add a custom collate function to `train.py` to detect empty boxes and exclude
* [x] Fix resizing transform by creating a custom `YoloResize` transform called `YoloResizeTransform`
* [x] Add finetuning to the `train.py` script
* [x] Fix the learning rate adjustment to decrease more consistently during training and finetuning
* [x] Created method to find optimal anchor box sizes with `kmeans.py` and a script to temporarily convert labels `scripts/convert_labels.py` (the converted labels are only used for calculating anchor values)
* [x] Ensure this codebase works with full sized YOLOv3 network
* [x] Fix `customloader.py` to take custom (as an argument) anchors, anchor numbers and model input dims
* [x] Clean up unnecessary params in config files
---
* [ ] Checkpoint only models with better loss values than previous ones (use checkpoint functionality in PyTorch)
* [ ] Ensure `live.py` is correctly drawing bounding boxes
* [ ] Ensure `eval.py` is correctly evaluating predictions
* [ ] flake8 (clean up extra blank lines, long lines, etc.)
* [ ] Remove `*` imports in place of explicit imports
